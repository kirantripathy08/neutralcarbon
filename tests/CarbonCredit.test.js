/**
 * neutralcarbon/tests/CarbonCredit.test.js
 * ------------------------------------------
 * Hardhat/Mocha/Chai test suite for CarbonCredit.sol
 *
 * Run: npx hardhat test tests/CarbonCredit.test.js
 */

const { expect }  = require("chai");
const { ethers }  = require("hardhat");
const { loadFixture } = require("@nomicfoundation/hardhat-toolbox/network-helpers");

describe("CarbonCredit", function () {

  // ── Fixture ────────────────────────────────────────────────────
  async function deployFixture() {
    const [owner, treasury, user1, user2, verifier] = await ethers.getSigners();
    const CarbonCredit = await ethers.getContractFactory("CarbonCredit");
    const cc = await CarbonCredit.deploy(treasury.address);
    return { cc, owner, treasury, user1, user2, verifier };
  }

  // ── Helpers ────────────────────────────────────────────────────
  const MINT_TONNES = 100n;
  const MINT_UNITS  = 100_000n;    // 1 unit = 1 kg → 100t = 100,000 units

  async function mintHelper(cc, user, tonnes = MINT_TONNES, units = MINT_UNITS) {
    const fee   = await cc.mintFeePerTonne();
    const value = fee * tonnes;
    const tx    = await cc.connect(user).mintCredit(
      "Brazil", "BRA", 2018, "mixed", tonnes, units, { value }
    );
    const receipt = await tx.wait();
    // Extract tokenId from CreditMinted event
    const event   = receipt.logs.find(
      l => l.fragment && l.fragment.name === "CreditMinted"
    );
    return event.args.tokenId;
  }

  // ── Deployment ─────────────────────────────────────────────────
  describe("Deployment", function () {
    it("sets the correct treasury address", async function () {
      const { cc, treasury } = await loadFixture(deployFixture);
      expect(await cc.treasury()).to.equal(treasury.address);
    });

    it("authorises the deployer as verifier", async function () {
      const { cc, owner } = await loadFixture(deployFixture);
      expect(await cc.verifiers(owner.address)).to.be.true;
    });

    it("starts with zero tokens and zero offset", async function () {
      const { cc } = await loadFixture(deployFixture);
      expect(await cc.totalTokens()).to.equal(0n);
      expect(await cc.totalOffsetTonnes()).to.equal(0n);
    });
  });

  // ── Minting ────────────────────────────────────────────────────
  describe("Minting", function () {
    it("mints a new token with correct metadata", async function () {
      const { cc, user1 } = await loadFixture(deployFixture);
      const tokenId = await mintHelper(cc, user1);

      const credit = await cc.getCredit(tokenId);
      expect(credit.country).to.equal("Brazil");
      expect(credit.year).to.equal(2018);
      expect(credit.tonnesCO2).to.equal(MINT_TONNES);
      expect(credit.status).to.equal(0n);    // PENDING
      expect(credit.minter).to.equal(user1.address);
    });

    it("assigns ERC-1155 balance to minter", async function () {
      const { cc, user1 } = await loadFixture(deployFixture);
      const tokenId = await mintHelper(cc, user1);
      expect(await cc.balanceOf(user1.address, tokenId)).to.equal(MINT_UNITS);
    });

    it("increments totalTokens", async function () {
      const { cc, user1 } = await loadFixture(deployFixture);
      await mintHelper(cc, user1);
      await mintHelper(cc, user1);
      expect(await cc.totalTokens()).to.equal(2n);
    });

    it("reverts if fee is insufficient", async function () {
      const { cc, user1 } = await loadFixture(deployFixture);
      await expect(
        cc.connect(user1).mintCredit("X", "XXX", 2018, "coal", 100n, 100000n, { value: 0n })
      ).to.be.revertedWith("CarbonCredit: insufficient mint fee");
    });

    it("reverts for invalid year", async function () {
      const { cc, user1 } = await loadFixture(deployFixture);
      const fee = (await cc.mintFeePerTonne()) * 10n;
      await expect(
        cc.connect(user1).mintCredit("X", "XXX", 1990, "coal", 10n, 10000n, { value: fee })
      ).to.be.revertedWith("CarbonCredit: invalid year");
    });

    it("forwards fee to treasury", async function () {
      const { cc, user1, treasury } = await loadFixture(deployFixture);
      const before = await ethers.provider.getBalance(treasury.address);
      const fee    = await cc.mintFeePerTonne();
      await mintHelper(cc, user1, MINT_TONNES, MINT_UNITS);
      const after  = await ethers.provider.getBalance(treasury.address);
      expect(after - before).to.equal(fee * MINT_TONNES);
    });
  });

  // ── Verification ───────────────────────────────────────────────
  describe("Verification", function () {
    it("allows verifier to verify a PENDING token", async function () {
      const { cc, user1, owner } = await loadFixture(deployFixture);
      const tokenId  = await mintHelper(cc, user1);
      const proofHash = ethers.keccak256(ethers.toUtf8Bytes("qml_report_1"));

      await cc.connect(owner).verifyCredit(tokenId, proofHash, "req-001");

      const credit = await cc.getCredit(tokenId);
      expect(credit.status).to.equal(1n);       // VERIFIED
      expect(credit.qmlProofHash).to.equal(proofHash);
    });

    it("updates totalOffsetKg on verification", async function () {
      const { cc, user1, owner } = await loadFixture(deployFixture);
      const tokenId   = await mintHelper(cc, user1, 50n, 50000n);
      const proofHash = ethers.keccak256(ethers.toUtf8Bytes("proof"));
      await cc.connect(owner).verifyCredit(tokenId, proofHash, "req-002");
      expect(await cc.totalOffsetTonnes()).to.equal(50n);
    });

    it("reverts if non-verifier tries to verify", async function () {
      const { cc, user1, user2 } = await loadFixture(deployFixture);
      const tokenId = await mintHelper(cc, user1);
      await expect(
        cc.connect(user2).verifyCredit(tokenId, ethers.ZeroHash, "req-x")
      ).to.be.revertedWith("CarbonCredit: caller is not a verifier");
    });

    it("cannot verify an already verified token", async function () {
      const { cc, user1, owner } = await loadFixture(deployFixture);
      const tokenId = await mintHelper(cc, user1);
      await cc.connect(owner).verifyCredit(tokenId, ethers.ZeroHash, "req-1");
      await expect(
        cc.connect(owner).verifyCredit(tokenId, ethers.ZeroHash, "req-2")
      ).to.be.revertedWith("CarbonCredit: not pending");
    });
  });

  // ── Revocation ─────────────────────────────────────────────────
  describe("Revocation", function () {
    it("verifier can revoke a PENDING token", async function () {
      const { cc, user1, owner } = await loadFixture(deployFixture);
      const tokenId = await mintHelper(cc, user1);
      await cc.connect(owner).revokeCredit(tokenId, "fraudulent emission data");
      expect((await cc.getCredit(tokenId)).status).to.equal(3n);   // REVOKED
    });

    it("revocation of VERIFIED token reduces totalOffset", async function () {
      const { cc, user1, owner } = await loadFixture(deployFixture);
      const tokenId = await mintHelper(cc, user1, 200n, 200000n);
      await cc.connect(owner).verifyCredit(tokenId, ethers.ZeroHash, "req-r");
      await cc.connect(owner).revokeCredit(tokenId, "audit failed");
      expect(await cc.totalOffsetTonnes()).to.equal(0n);
    });
  });

  // ── Retirement ─────────────────────────────────────────────────
  describe("Retirement", function () {
    it("holder can retire (burn) verified credits", async function () {
      const { cc, user1, owner } = await loadFixture(deployFixture);
      const tokenId = await mintHelper(cc, user1, 100n, 100000n);
      await cc.connect(owner).verifyCredit(tokenId, ethers.ZeroHash, "req-ret");

      await cc.connect(user1).retireCredit(tokenId, 50000n);
      expect(await cc.balanceOf(user1.address, tokenId)).to.equal(50000n);
    });

    it("full retirement marks status as RETIRED", async function () {
      const { cc, user1, owner } = await loadFixture(deployFixture);
      const tokenId = await mintHelper(cc, user1, 10n, 10000n);
      await cc.connect(owner).verifyCredit(tokenId, ethers.ZeroHash, "req-full");
      await cc.connect(user1).retireCredit(tokenId, 10000n);
      expect((await cc.getCredit(tokenId)).status).to.equal(2n);   // RETIRED
    });

    it("cannot retire unverified credit", async function () {
      const { cc, user1 } = await loadFixture(deployFixture);
      const tokenId = await mintHelper(cc, user1);
      await expect(
        cc.connect(user1).retireCredit(tokenId, 1000n)
      ).to.be.revertedWith("CarbonCredit: credit not verified");
    });
  });

  // ── Access Control ─────────────────────────────────────────────
  describe("Access control", function () {
    it("owner can add and remove verifiers", async function () {
      const { cc, owner, verifier } = await loadFixture(deployFixture);
      await cc.connect(owner).addVerifier(verifier.address);
      expect(await cc.verifiers(verifier.address)).to.be.true;
      await cc.connect(owner).removeVerifier(verifier.address);
      expect(await cc.verifiers(verifier.address)).to.be.false;
    });

    it("non-owner cannot add verifier", async function () {
      const { cc, user1, verifier } = await loadFixture(deployFixture);
      await expect(
        cc.connect(user1).addVerifier(verifier.address)
      ).to.be.reverted;
    });
  });
});
