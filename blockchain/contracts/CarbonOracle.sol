// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title  CarbonOracle
 * @notice Chainlink oracle consumer that bridges off-chain QML anomaly detection
 *         results to the on-chain CarbonCredit smart contract.
 *
 * Flow:
 *   1. Backend runs QML anomaly detection on emission data
 *   2. Backend calls requestVerification() with country + year
 *   3. Chainlink Any-API job fetches the QML result from the off-chain API
 *   4. fulfill() is called with the verification result
 *   5. If verified, CarbonCredit.verifyCredit() is called
 *
 * References:
 *   Chainlink Any-API: https://docs.chain.link/any-api/introduction
 *   Chainlink Direct Requests: https://docs.chain.link/chainlink-nodes
 *
 * Authors: Kiran Tripathy · Kolangada Advaith Dilip — VIT 2026
 */

import "@chainlink/contracts/src/v0.8/ChainlinkClient.sol";
import "@chainlink/contracts/src/v0.8/ConfirmedOwner.sol";
import "./CarbonCredit.sol";

contract CarbonOracle is ChainlinkClient, ConfirmedOwner {
    using Chainlink for Chainlink.Request;

    // ────────────────────────────────────────────────────────────
    // State
    // ────────────────────────────────────────────────────────────

    CarbonCredit public immutable carbonCredit;

    // Chainlink oracle config
    address private oracle;
    bytes32 private jobId;
    uint256 private fee;             // LINK payment per request

    // Pending requests: Chainlink requestId → pending verification
    struct PendingVerification {
        uint256 tokenId;
        string  country;
        uint16  year;
        bool    exists;
    }
    mapping(bytes32 => PendingVerification) public pendingRequests;

    // QML API endpoint (set at deploy time, updateable)
    string public qmlApiBaseUrl;

    // Events
    event VerificationRequested(bytes32 indexed requestId, uint256 tokenId);
    event VerificationFulfilled(bytes32 indexed requestId, uint256 tokenId, bool approved);
    event OracleConfigUpdated(address oracle, bytes32 jobId, uint256 fee);

    // ────────────────────────────────────────────────────────────
    // Constructor
    // ────────────────────────────────────────────────────────────

    /**
     * @param _carbonCredit  Address of deployed CarbonCredit contract
     * @param _linkToken     LINK token address (network-specific)
     * @param _oracle        Chainlink oracle address
     * @param _jobId         Chainlink job ID for Any-API GET request
     * @param _feeLINK       Fee in LINK (e.g. 0.1 LINK = 100000000000000000)
     * @param _qmlApiUrl     Base URL of the QML verification API
     */
    constructor(
        address _carbonCredit,
        address _linkToken,
        address _oracle,
        bytes32 _jobId,
        uint256 _feeLINK,
        string memory _qmlApiUrl
    ) ConfirmedOwner(msg.sender) {
        setChainlinkToken(_linkToken);
        carbonCredit  = CarbonCredit(_carbonCredit);
        oracle        = _oracle;
        jobId         = _jobId;
        fee           = _feeLINK;
        qmlApiBaseUrl = _qmlApiUrl;
    }

    // ────────────────────────────────────────────────────────────
    // Oracle Request
    // ────────────────────────────────────────────────────────────

    /**
     * @notice Request QML verification for a carbon credit token.
     *         Sends a Chainlink request to the QML API and maps the
     *         requestId to the token being verified.
     *
     * @param tokenId   The ERC-1155 token ID to verify
     * @param country   Country name (for API query)
     * @param year      Emission year (for API query)
     */
    function requestVerification(
        uint256 tokenId,
        string memory country,
        uint16 year
    ) external onlyOwner returns (bytes32 requestId) {
        // Build Chainlink request
        Chainlink.Request memory req = buildChainlinkRequest(
            jobId,
            address(this),
            this.fulfill.selector
        );

        // Construct API URL:
        // e.g. https://api.neutralcarbon.io/verify?country=China&year=2018
        string memory url = string(abi.encodePacked(
            qmlApiBaseUrl,
            "?country=", country,
            "&year=", _uint16ToString(year),
            "&tokenId=", _uintToString(tokenId)
        ));
        req.add("get", url);

        // Parse JSON path: { "verified": true, "qml_score": 0.92, "proof_hash": "0x..." }
        req.add("path", "verified");

        requestId = sendChainlinkRequestTo(oracle, req, fee);

        pendingRequests[requestId] = PendingVerification({
            tokenId: tokenId,
            country: country,
            year:    year,
            exists:  true
        });

        emit VerificationRequested(requestId, tokenId);
        return requestId;
    }

    // ────────────────────────────────────────────────────────────
    // Oracle Callback
    // ────────────────────────────────────────────────────────────

    /**
     * @notice Callback from Chainlink oracle with verification result.
     * @param _requestId  The Chainlink request ID
     * @param _verified   true if QML approved, false if flagged as fraud
     */
    function fulfill(bytes32 _requestId, bool _verified)
        external
        recordChainlinkFulfillment(_requestId)
    {
        PendingVerification storage pv = pendingRequests[_requestId];
        require(pv.exists, "CarbonOracle: unknown request");

        if (_verified) {
            // Create a pseudo proof hash from the request ID + block hash
            bytes32 proofHash = keccak256(abi.encodePacked(
                _requestId,
                blockhash(block.number - 1),
                pv.country,
                pv.year
            ));
            carbonCredit.verifyCredit(
                pv.tokenId,
                proofHash,
                _bytes32ToString(_requestId)
            );
        }

        emit VerificationFulfilled(_requestId, pv.tokenId, _verified);
        delete pendingRequests[_requestId];
    }

    // ────────────────────────────────────────────────────────────
    // Admin
    // ────────────────────────────────────────────────────────────

    function updateOracleConfig(
        address _oracle,
        bytes32 _jobId,
        uint256 _fee
    ) external onlyOwner {
        oracle = _oracle;
        jobId  = _jobId;
        fee    = _fee;
        emit OracleConfigUpdated(_oracle, _jobId, _fee);
    }

    function updateQmlApiUrl(string memory _url) external onlyOwner {
        qmlApiBaseUrl = _url;
    }

    function withdrawLink() external onlyOwner {
        LinkTokenInterface link = LinkTokenInterface(chainlinkTokenAddress());
        require(link.transfer(msg.sender, link.balanceOf(address(this))),
                "CarbonOracle: LINK transfer failed");
    }

    // ────────────────────────────────────────────────────────────
    // Internal helpers
    // ────────────────────────────────────────────────────────────

    function _uint16ToString(uint16 v) internal pure returns (string memory) {
        if (v == 0) return "0";
        uint16 tmp = v;
        uint8 digits;
        while (tmp != 0) { digits++; tmp /= 10; }
        bytes memory buf = new bytes(digits);
        while (v != 0) { buf[--digits] = bytes1(uint8(48 + v % 10)); v /= 10; }
        return string(buf);
    }

    function _uintToString(uint256 v) internal pure returns (string memory) {
        if (v == 0) return "0";
        uint256 tmp = v;
        uint8 digits;
        while (tmp != 0) { digits++; tmp /= 10; }
        bytes memory buf = new bytes(digits);
        while (v != 0) { buf[--digits] = bytes1(uint8(48 + v % 10)); v /= 10; }
        return string(buf);
    }

    function _bytes32ToString(bytes32 b) internal pure returns (string memory) {
        bytes memory alphabet = "0123456789abcdef";
        bytes memory str = new bytes(2 + 64);
        str[0] = "0"; str[1] = "x";
        for (uint256 i = 0; i < 32; i++) {
            str[2 + i * 2]     = alphabet[uint8(b[i] >> 4)];
            str[3 + i * 2]     = alphabet[uint8(b[i] & 0x0f)];
        }
        return string(str);
    }
}
