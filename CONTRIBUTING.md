# Contributing Guidelines

## Code Review Process

The AI Gateway uses a streamlined review process optimized for speed and innovation, differing from the standard GitLab [code review guidelines](https://docs.gitlab.com/development/code_review/).

### Review Requirements

**For AI Gateway and Duo Workflow Service:**

1. **Reviewers**: All AI Engineering department engineers are reviewers (no prerequisites required).
1. **Maintainers** All Senior+ Backend, Full Stack, and Machine learning engineers in AI Engineering are expected to be maintainers, unless they are already maintaining multiple other projects.
1. **Reviews**: Requesting a review from a reviewer before a maintainer is suggested to give more reviewers experience. If you have a particularly time sensitive MR, you can go straight to a maintainer.
1. **Deployment**: Merged MRs automatically deploy with [Runway](https://handbook.gitlab.com/handbook/engineering/infrastructure/platforms/tools/runway/)
1. **Domain expertise**:
   - Recommendations encoded in [Dangerfile](Dangerfile)
   - Requirements encoded in [CODEOWNERS](.gitlab/CODEOWNERS)

### Review Guidelines

**Required practices:**

- Authors cannot approve their own MRs
- Never merge code you don't understand or aren't confident in
- By default, Pylint/mypy rules should not be disabled inline because it negates agreed-upon code standards that the
rule is attempting to apply to the codebase. If you must use inline disable, provide the reason as a code comment where
the rule is disabled.

**Always request a second review when:**

- Changing unfamiliar code areas
- Touching complex areas (authentication, authorization, etc.)
- You want additional validation
- The author requests multiple review rounds

The streamlined process balances speed with quality, supporting rapid AI feature development while maintaining code standards.
