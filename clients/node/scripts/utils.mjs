import { readFile, access } from "node:fs/promises";
import { join } from "node:path";
import { createHash } from "node:crypto";

// Files to hash - relative to project root
export const FILE_TO_HASH = "src/grpc/contract.ts";

// Hash file location
export const HASH_FILE = join(process.cwd(), ".file-hash");

export async function calculateHash() {
  const hash = createHash("md5");

  const filePath = join(process.cwd(), FILE_TO_HASH);
  try {
    const content = await readFile(filePath, "utf8");
    hash.update(content);
  } catch (err) {
    console.warn(`Couldn't read file ${filePath}:`, err.message);
  }

  return hash.digest("hex");
}

/**
 * Check if hash file exists
 */
export async function hashFileExists() {
  try {
    await access(HASH_FILE);
    return true;
  } catch {
    return false;
  }
}
