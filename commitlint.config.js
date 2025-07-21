module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    "subject-case": [
      0,
      "always",
      "lowerCase"
    ],
    "body-max-line-length": [0, "always", 150]
  },
};
