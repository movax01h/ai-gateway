#!/usr/bin/env node

import { writeFile } from "node:fs/promises";

import { calculateHash, HASH_FILE } from "./utils.mjs";

export async function storeHash(hash) {
  await writeFile(HASH_FILE, hash);
  return hash;
}

async function main() {
  try {
    const hash = await calculateHash();
    await storeHash(hash);
    console.log(`Stored hash of generated file: ${hash}`);
  } catch (err) {
    console.error("Error storing hash:", err);
    process.exit(1);
  }
}

main();
