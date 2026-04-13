/**
 * neutralcarbon/blockchain/scripts/deploy.js
 * -------------------------------------------
 * Deploys CarbonCredit and CarbonOracle contracts.
 *
 * Usage:
 *   npx hardhat run scripts/deploy.js --network localhost
 *   npx hardhat run scripts/deploy.js --network sepolia
 */

const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

// Chainlink config per network
const CHAINLINK_CONFIG = {
  localhost: {
    linkToken: "0x0000000000000000000000000000000000000001", // mock
    oracle:    "0x0000000000000000000000000000000000000002", // mock
    jobId:     ethers.encodeBytes32String("mock-job-id"),
    fee:       ethers.parseEther("0.1"),                    // 0.1 LINK
  },
  sepolia: {
    linkToken: "0x779877A7B0D9E8603169DdbD7836e478b4624789",
    oracle:    "0x6090149792dAAeE9D1D568c9f9a6F6B46AA29DF",
    jobId:     ethers.encodeBytes32String("ca98366cc7314957b8c012c72f05aeeb"),
    fee:       ethers.parseEther("0.1"),
  },
};

async function main() {
  const network = hre.network.name;
  console.log(`\n${"=".repeat(60)}`);
  console.log(`NeutralCarbon — Deploying to: ${network}`);
  console.log(`${"=".repeat(60)}\n`);

  const [deployer, treasury] = await ethers.getSigners();
  console.log(`Deployer:  ${deployer.address}`);
  console.log(`Treasury:  ${treasury ? treasury.address : deployer.address}`);
  const treasuryAddr = treasury ? treasury.address : deployer.address;

  const balance = await ethers.provider.getBalance(deployer.address);
  console.log(`Balance:   ${ethers.formatEther(balance)} ETH\n`);

  // ── 1. Deploy CarbonCredit ───────────────────────────────────
  console.log("Deploying CarbonCredit (ERC-1155)…");
  const CarbonCredit = await ethers.getContractFactory("CarbonCredit");
  const carbonCredit = await CarbonCredit.deploy(treasuryAddr);
  await carbonCredit.waitForDeployment();
  const ccAddress = await carbonCredit.getAddress();
  console.log(`✓  CarbonCredit deployed → ${ccAddress}`);

  // ── 2. Deploy CarbonOracle ───────────────────────────────────
  const cfg = CHAINLINK_CONFIG[network] || CHAINLINK_CONFIG.localhost;
  const QML_API_URL = process.env.QML_API_URL || "https://api.neutralcarbon.io/verify";

  console.log("\nDeploying CarbonOracle (Chainlink consumer)…");
  const CarbonOracle = await ethers.getContractFactory("CarbonOracle");
  const carbonOracle = await CarbonOracle.deploy(
    ccAddress,
    cfg.linkToken,
    cfg.oracle,
    cfg.jobId,
    cfg.fee,
    QML_API_URL
  );
  await carbonOracle.waitForDeployment();
  const oracleAddress = await carbonOracle.getAddress();
  console.log(`✓  CarbonOracle deployed → ${oracleAddress}`);

  // ── 3. Add oracle as verifier ────────────────────────────────
  console.log("\nAuthorising oracle as verifier…");
  const tx = await carbonCredit.addVerifier(oracleAddress);
  await tx.wait();
  console.log(`✓  Oracle authorised as verifier`);

  // ── 4. Mint sample credits for demo ─────────────────────────
  if (network === "localhost") {
    console.log("\nMinting sample demo credits…");
    const samples = [
      { country: "Brazil",  code: "BRA", year: 2018, source: "mixed",  tonnes: 2400, units: 2400000 },
      { country: "Germany", code: "DEU", year: 2018, source: "coal",   tonnes: 1100, units: 1100000 },
      { country: "Canada",  code: "CAN", year: 2017, source: "oil",    tonnes:  850, units:  850000 },
    ];
    const mintFee = await carbonCredit.mintFeePerTonne();
    for (const s of samples) {
      const value = mintFee * BigInt(s.tonnes);
      const tx = await carbonCredit.mintCredit(
        s.country, s.code, s.year, s.source, s.tonnes, s.units,
        { value }
      );
      const receipt = await tx.wait();
      console.log(`  ✓  Minted ${s.country} (${s.tonnes}t) — tx: ${receipt.hash.slice(0,10)}…`);
    }
  }

  // ── 5. Save deployment info ──────────────────────────────────
  const deployInfo = {
    network,
    deployedAt: new Date().toISOString(),
    deployer:   deployer.address,
    contracts: {
      CarbonCredit: ccAddress,
      CarbonOracle: oracleAddress,
    },
    chainlink: {
      linkToken: cfg.linkToken,
      oracle:    cfg.oracle,
    },
  };

  const outPath = path.join(__dirname, `../deployments/${network}.json`);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(deployInfo, null, 2));
  console.log(`\n✓  Deployment info saved → ${outPath}`);

  console.log(`\n${"=".repeat(60)}`);
  console.log("Deployment complete!");
  console.log(`  CarbonCredit : ${ccAddress}`);
  console.log(`  CarbonOracle : ${oracleAddress}`);
  console.log(`${"=".repeat(60)}\n`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
