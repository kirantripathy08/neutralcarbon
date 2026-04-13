// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title  CarbonCredit
 * @notice ERC-1155 semi-fungible carbon credit token for the NeutralCarbon platform.
 *
 * Token IDs:
 *   Each unique (country, year, source) combination maps to a distinct token ID.
 *   Fractionalisation is supported — 1 token unit = 1 kg CO₂ offset (1000 units = 1 tonne).
 *
 * Lifecycle:
 *   PENDING  → verified by QML oracle → VERIFIED  → retired by holder → RETIRED
 *                                    ↘ rejected                       → REVOKED
 *
 * References:
 *   ERC-1155 Multi-Token Standard: https://ethereum.org/en/developers/docs/standards/tokens/erc-1155/
 *   Chainlink Oracle integration: CarbonOracle.sol
 *
 * Authors: Kiran Tripathy · Kolangada Advaith Dilip — VIT 2026
 */

import "@openzeppelin/contracts/token/ERC1155/ERC1155.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract CarbonCredit is ERC1155, Ownable, ReentrancyGuard {
    using Counters for Counters.Counter;

    // ────────────────────────────────────────────────────────────
    // Enums & Structs
    // ────────────────────────────────────────────────────────────

    enum TokenStatus { PENDING, VERIFIED, RETIRED, REVOKED }

    struct CreditMetadata {
        string  country;          // ISO country name
        string  countryCode;      // ISO-3 code (e.g. "CHN")
        uint16  year;             // Emission year
        string  emissionSource;   // "coal" | "oil" | "gas" | "cement" | "mixed"
        uint256 tonnesCO2;        // Total CO₂ offset in tonnes
        TokenStatus status;
        uint256 mintTimestamp;
        uint256 verifyTimestamp;
        address minter;
        bytes32 qmlProofHash;     // SHA-256 hash of off-chain QML verification report
        string  oracleRequestId;  // Chainlink request ID for traceability
    }

    // ────────────────────────────────────────────────────────────
    // State
    // ────────────────────────────────────────────────────────────

    Counters.Counter private _tokenIds;

    mapping(uint256 => CreditMetadata) public credits;

    // Authorised verifier addresses (oracle contract + admin multi-sig)
    mapping(address => bool) public verifiers;

    // Country → list of token IDs
    mapping(string => uint256[]) public creditsByCountry;

    // Total CO₂ offset tracked by the protocol (kg)
    uint256 public totalOffsetKg;

    // Fee (wei) charged per tonne at mint time — sent to treasury
    uint256 public mintFeePerTonne = 0.001 ether;
    address public treasury;

    // ────────────────────────────────────────────────────────────
    // Events
    // ────────────────────────────────────────────────────────────

    event CreditMinted(
        uint256 indexed tokenId,
        address indexed minter,
        string  country,
        uint16  year,
        uint256 tonnesCO2
    );
    event CreditVerified(uint256 indexed tokenId, bytes32 qmlProofHash);
    event CreditRevoked(uint256 indexed tokenId, string reason);
    event CreditRetired(uint256 indexed tokenId, address indexed retiree, uint256 amount);
    event VerifierAdded(address indexed verifier);
    event VerifierRemoved(address indexed verifier);

    // ────────────────────────────────────────────────────────────
    // Modifiers
    // ────────────────────────────────────────────────────────────

    modifier onlyVerifier() {
        require(verifiers[msg.sender], "CarbonCredit: caller is not a verifier");
        _;
    }

    modifier tokenExists(uint256 tokenId) {
        require(credits[tokenId].mintTimestamp != 0, "CarbonCredit: token does not exist");
        _;
    }

    // ────────────────────────────────────────────────────────────
    // Constructor
    // ────────────────────────────────────────────────────────────

    constructor(address _treasury)
        ERC1155("https://neutralcarbon.io/api/token/{id}.json")
    {
        treasury = _treasury;
        verifiers[msg.sender] = true;   // deployer is initial verifier
    }

    // ────────────────────────────────────────────────────────────
    // Minting
    // ────────────────────────────────────────────────────────────

    /**
     * @notice Mint a new carbon credit token.
     * @param  country         ISO country name
     * @param  countryCode     ISO-3 code
     * @param  year            Emission year (1992–2100)
     * @param  emissionSource  "coal" | "oil" | "gas" | "cement" | "mixed"
     * @param  tonnesCO2       CO₂ offset quantity in tonnes
     * @param  amount          Token units to mint (1 unit = 1 kg = 1/1000 tonne)
     * @return tokenId         Newly minted token ID
     */
    function mintCredit(
        string  memory country,
        string  memory countryCode,
        uint16         year,
        string  memory emissionSource,
        uint256        tonnesCO2,
        uint256        amount
    ) external payable nonReentrant returns (uint256 tokenId) {
        require(bytes(country).length > 0,    "CarbonCredit: empty country");
        require(tonnesCO2 > 0,                "CarbonCredit: zero offset");
        require(amount > 0,                   "CarbonCredit: zero amount");
        require(year >= 1992 && year <= 2100, "CarbonCredit: invalid year");
        require(
            msg.value >= mintFeePerTonne * tonnesCO2,
            "CarbonCredit: insufficient mint fee"
        );

        _tokenIds.increment();
        tokenId = _tokenIds.current();

        credits[tokenId] = CreditMetadata({
            country:         country,
            countryCode:     countryCode,
            year:            year,
            emissionSource:  emissionSource,
            tonnesCO2:       tonnesCO2,
            status:          TokenStatus.PENDING,
            mintTimestamp:   block.timestamp,
            verifyTimestamp: 0,
            minter:          msg.sender,
            qmlProofHash:    bytes32(0),
            oracleRequestId: ""
        });

        creditsByCountry[country].push(tokenId);

        _mint(msg.sender, tokenId, amount, "");

        // Forward fee to treasury
        (bool sent, ) = treasury.call{value: msg.value}("");
        require(sent, "CarbonCredit: fee transfer failed");

        emit CreditMinted(tokenId, msg.sender, country, year, tonnesCO2);
        return tokenId;
    }

    // ────────────────────────────────────────────────────────────
    // Verification (called by CarbonOracle or admin)
    // ────────────────────────────────────────────────────────────

    /**
     * @notice Mark a token as VERIFIED after QML + oracle confirmation.
     * @param  tokenId       Token to verify
     * @param  qmlProofHash  SHA-256 of the QML anomaly detection report
     * @param  oracleReqId   Chainlink requestId for audit trail
     */
    function verifyCredit(
        uint256        tokenId,
        bytes32        qmlProofHash,
        string memory  oracleReqId
    ) external onlyVerifier tokenExists(tokenId) {
        CreditMetadata storage c = credits[tokenId];
        require(c.status == TokenStatus.PENDING, "CarbonCredit: not pending");

        c.status          = TokenStatus.VERIFIED;
        c.verifyTimestamp = block.timestamp;
        c.qmlProofHash    = qmlProofHash;
        c.oracleRequestId = oracleReqId;

        totalOffsetKg += c.tonnesCO2 * 1000;

        emit CreditVerified(tokenId, qmlProofHash);
    }

    /**
     * @notice Revoke a fraudulent or erroneous credit.
     */
    function revokeCredit(
        uint256       tokenId,
        string memory reason
    ) external onlyVerifier tokenExists(tokenId) {
        CreditMetadata storage c = credits[tokenId];
        require(
            c.status != TokenStatus.RETIRED,
            "CarbonCredit: cannot revoke retired credit"
        );
        // Reduce tracked offset if it was verified
        if (c.status == TokenStatus.VERIFIED) {
            totalOffsetKg -= c.tonnesCO2 * 1000;
        }
        c.status = TokenStatus.REVOKED;
        emit CreditRevoked(tokenId, reason);
    }

    // ────────────────────────────────────────────────────────────
    // Retirement (burn)
    // ────────────────────────────────────────────────────────────

    /**
     * @notice Retire (burn) `amount` units of a verified credit.
     *         Retired credits cannot be re-used (prevents double-counting).
     */
    function retireCredit(
        uint256 tokenId,
        uint256 amount
    ) external nonReentrant tokenExists(tokenId) {
        CreditMetadata storage c = credits[tokenId];
        require(c.status == TokenStatus.VERIFIED, "CarbonCredit: credit not verified");
        require(
            balanceOf(msg.sender, tokenId) >= amount,
            "CarbonCredit: insufficient balance"
        );

        _burn(msg.sender, tokenId, amount);

        // If all units burned, mark as retired
        if (totalSupply(tokenId) == 0) {
            c.status = TokenStatus.RETIRED;
        }

        emit CreditRetired(tokenId, msg.sender, amount);
    }

    // ────────────────────────────────────────────────────────────
    // View helpers
    // ────────────────────────────────────────────────────────────

    function getCredit(uint256 tokenId)
        external view tokenExists(tokenId)
        returns (CreditMetadata memory)
    {
        return credits[tokenId];
    }

    function getCreditsByCountry(string memory country)
        external view
        returns (uint256[] memory)
    {
        return creditsByCountry[country];
    }

    function totalTokens() external view returns (uint256) {
        return _tokenIds.current();
    }

    function totalOffsetTonnes() external view returns (uint256) {
        return totalOffsetKg / 1000;
    }

    // ────────────────────────────────────────────────────────────
    // Admin
    // ────────────────────────────────────────────────────────────

    function addVerifier(address verifier) external onlyOwner {
        verifiers[verifier] = true;
        emit VerifierAdded(verifier);
    }

    function removeVerifier(address verifier) external onlyOwner {
        verifiers[verifier] = false;
        emit VerifierRemoved(verifier);
    }

    function setMintFee(uint256 feePerTonne) external onlyOwner {
        mintFeePerTonne = feePerTonne;
    }

    function setURI(string memory newUri) external onlyOwner {
        _setURI(newUri);
    }

    // ERC-1155 supply tracking (required for retirement check)
    mapping(uint256 => uint256) private _totalSupply;

    function totalSupply(uint256 tokenId) public view returns (uint256) {
        return _totalSupply[tokenId];
    }

    function _beforeTokenTransfer(
        address operator,
        address from,
        address to,
        uint256[] memory ids,
        uint256[] memory amounts,
        bytes memory data
    ) internal override {
        super._beforeTokenTransfer(operator, from, to, ids, amounts, data);
        for (uint256 i = 0; i < ids.length; i++) {
            if (from == address(0)) _totalSupply[ids[i]] += amounts[i];
            if (to   == address(0)) _totalSupply[ids[i]] -= amounts[i];
        }
    }
}
