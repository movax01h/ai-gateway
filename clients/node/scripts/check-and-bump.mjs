#!/usr/bin/env node

import { readFile, unlink } from "node:fs/promises";
import { exec } from "node:child_process";
import { promisify } from "node:util";
import { calculateHash, HASH_FILE, hashFileExists } from "./utils.mjs";

const execAsync = promisify(exec);

async function removeHashFile() {
  try {
    await unlink(HASH_FILE);
  } catch (err) {
    if (err.code !== "ENOENT") {
      console.error(
        `Error removing hash file "${HASH_FILE}", please remove it manually`,
        err
      );
    }
  }
}

async function main() {
  try {
    if (!(await hashFileExists())) {
      throw new Error(
        "No previous hash file found. This probably means something went wrong. Node client version will not be incremented, please manually verify!"
      );
    }

    const currentHash = await calculateHash();
    const previousHash = await readFile(HASH_FILE, "utf8");

    if (currentHash !== previousHash) {
      console.log(
        "Node client files seem to have changed! Bumping package version..."
      );
      try {
        const { stdout } = await execAsync(
          "npm version patch --no-git-tag-version"
        );
        console.log("Version bumped:", stdout.trim());
      } catch (err) {
        console.error("Failed to bump version:", err);
        process.exit(1);
      }
    } else {
      console.debug(
        "No changes detected to Node client. Version remains the same."
      );
    }
  } catch (err) {
    console.error("Error checking/bumping version:", err);
    process.exit(1);
  } finally {
    removeHashFile();
  }
}

main();
