/**
 * neutralcarbon/blockchain/scripts/mint.js
 * ------------------------------------------
 * Mints carbon credit tokens from a JSON input file or CLI args.
 *
 * Usage:
 *   # Mint from JSON batch file:
 *   npx hardhat run scripts/mint.js --network localhost
 *
 *   # Or pass env vars:
 *   COUNTRY="India" YEAR=2018 TONNES=500 \
 *     npx hardhat run scripts/mint.js --network localhost
 */

const { ethers } = require("hardhat");
const fs  = require("fs");
const path = require("path");

// ── Load deployment address ──────────────────────────────────────────────
function loadDeployment(network) {
  const p = path.join(__dirname, `../deployments/${network}.json`);
  if (!fs.existsSync(p)) {
    throw new Error(
      `No deployment found for network "${network}".\n` +
      `Run: npx hardhat run scripts/deploy.js --network ${network}`
    );
  }
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

// ── Default batch: top 10 emitters for demo ──────────────────────────────
const DEFAULT_BATCH = [
  { country: "China",        code: "CHN", year: 2018, source: "coal",   tonnes: 9297 },
  { country: "United States",code: "USA", year: 2018, source: "mixed",  tonnes: 5491 },
  { country: "India",        code: "IND", year: 2018, source: "coal",   tonnes: 2446 },
  { country: "Russia",       code: "RUS", year: 2018, source: "gas",    tonnes: 1327 },
  { country: "Japan",        code: "JPN", year: 2018, source: "mixed",  tonnes: 1011 },
  { country: "Germany",      code: "DEU", year: 2018, source: "coal",   tonnes:  510 },
  { country: "South Korea",  code: "KOR", year: 2018, source: "coal",   tonnes:  526 },
  { country: "Iran",         code: "IRN", year: 2018, source: "gas",    tonnes:  452 },
  { country: "Saudi Arabia", code: "SAU", year: 2018, source: "oil",    tonnes:  462 },
  { country: "Canada",       code: "CAN", year: 2018, source: "oil",    tonnes:  470 },
];

// ── Helper: format token ID as hex ──────────────────────────────────────
function fmtId(id) {
  return `0x${id.toString(16).padStart(4, "0")}`;
}

async function main() {
  const network = hre.network.name;
  const deploy  = loadDeployment(network);
  const ccAddr  = deploy.contracts.CarbonCredit;

  console.log(`\n${"=".repeat(60)}`);
  console.log(`NeutralCarbon — Minting Carbon Credits`);
  console.log(`Network:  ${network}`);
  console.log(`Contract: ${ccAddr}`);
  console.log(`${"=".repeat(60)}\n`);

  const [minter] = await ethers.getSigners();
  const cc = await ethers.getContractAt("CarbonCredit", ccAddr, minter);

  // Check fee per tonne
  const feePerTonne = await cc.mintFeePerTonne();
  console.log(`Mint fee: ${ethers.formatEther(feePerTonne)} ETH/tonne\n`);

  // Determine batch
  let batch;
  if (process.env.COUNTRY) {
    batch = [{
      country: process.env.COUNTRY,
      code:    process.env.CODE    || "XXX",
      year:    parseInt(process.env.YEAR)   || 2018,
      source:  process.env.SOURCE  || "mixed",
      tonnes:  parseInt(process.env.TONNES) || 100,
    }];
  } else if (process.env.BATCH_FILE && fs.existsSync(process.env.BATCH_FILE)) {
    batch = JSON.parse(fs.readFileSync(process.env.BATCH_FILE, "utf8"));
  } else {
    batch = DEFAULT_BATCH;
  }

  console.log(`Minting ${batch.length} tokens…\n`);

  const results = [];

  for (const item of batch) {
    const { country, code, year, source, tonnes } = item;
    const units = tonnes * 1000;                     // 1 unit = 1 kg
    const value = feePerTonne * BigInt(tonnes);

    try {
      const tx = await cc.mintCredit(
        country, code, year, source, tonnes, units, { value }
      );
      const receipt = await tx.wait();

      // Extract tokenId from CreditMinted event
      const event   = receipt.logs.find(
        l => l.fragment && l.fragment.name === "CreditMinted"
      );
      const tokenId = event ? event.args.tokenId : BigInt(0);

      const info = {
        tokenId:  fmtId(tokenId),
        country,
        year,
        source,
        tonnes,
        units,
        txHash:  receipt.hash,
        status:  "PENDING",
      };
      results.push(info);

      console.log(
        `  ✓ ${fmtId(tokenId)} | ${country.padEnd(15)} | ${year} | ` +
        `${tonnes.toString().padStart(6)}t | tx: ${receipt.hash.slice(0, 12)}…`
      );
    } catch (err) {
      console.error(`  ✗ ${country} (${year}): ${err.message.split("\n")[0]}`);
      results.push({ country, year, error: err.message });
    }
  }

  // Save mint results
  const outPath = path.join(
    __dirname,
    `../deployments/minted_${network}_${Date.now()}.json`
  );
  fs.writeFileSync(outPath, JSON.stringify(results, null, 2));

  console.log(`\n${"=".repeat(60)}`);
  console.log(`Minted: ${results.filter(r => !r.error).length}/${batch.length}`);
  console.log(`Results saved → ${outPath}`);
  console.log(`${"=".repeat(60)}\n`);

  // Print summary table
  const verified = results.filter(r => r.tokenId);
  if (verified.length > 0) {
    const totalTonnes = verified.reduce((s, r) => s + r.tonnes, 0);
    console.log(`Total CO₂ offset tokenized: ${totalTonnes.toLocaleString()} tonnes`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
