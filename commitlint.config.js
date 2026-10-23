module.exports = {
  extends: ['@commitlint/config-conventional'],
  ignores: [
    // GitLab generates this subject when a suggestion is applied in the UI. MRs squash, so the
    // generated commit never reaches main and the MR title is linted on its own.
    (message) => /^Apply \d+ suggestion\(s\) to \d+ file\(s\)/.test(message),
  ],
  rules: {
    "subject-case": [
      0,
      "always",
      "lowerCase"
    ],
    "body-max-line-length": [0, "always", 150],
    "footer-max-line-length": [0, "always", 150]
  },
};
