# Project Activity Flow

## What It Does

This flow automatically generates a comprehensive project activity summary for any specified date range by collecting and analyzing issues, merge requests, and project activity data. It creates a summary issue that provides insights into project progress and team contributions over the specified period.

## When to Use

- Generating project activity reports for any custom date range (daily, weekly, monthly, quarterly, etc.)
- Creating automated project activity summaries for stakeholders
- Tracking project progress and team activity over specific time periods
- Providing visibility into issue resolution and merge request activity
- Creating retrospective summaries for sprint reviews or milestone completions

## How It Works

The flow consists of eight sequential steps that comprehensively analyze project activity:

1. **Fetch New Issues**: Retrieves issues created during the specified date range
1. **Fetch Closed Issues**: Collects issues that were resolved and closed during the period
1. **Fetch Updated Issues**: Gathers issues that had significant activity or updates during the period
1. **Fetch New Merge Requests**: Retrieves merge requests opened during the specified date range
1. **Fetch Closed Merge Requests**: Collects merge requests that were successfully merged (not just closed) during the period
1. **Fetch Updated Merge Requests**: Gathers merge requests that had significant activity during the period
1. **Summarize Activity**: Analyzes all collected data to generate a comprehensive activity digest with comment summaries and collaboration highlights
1. **Create Summary Issue**: Creates a new issue titled "Activity Digest - [date range]" containing the complete activity summary

## What's Included in the Summary

The generated activity digest provides a comprehensive overview structured as follows:

### Issue Analysis (by category)

- **New Issues**: Issues opened during the specified period with descriptions and current status
- **Closed Issues**: Issues resolved and closed, including resolution details
- **Updated Issues**: Existing issues with meaningful activity, discussions, or progress updates
- **Comment Activity**: Summarized conversations and decisions from issue discussions

### Merge Request Analysis (by category)

- **New Merge Requests**: MRs opened during the period with descriptions and current status
- **Merged Requests**: Successfully merged MRs highlighting delivered functionality
- **Updated Merge Requests**: MRs with review activity, code discussions, or significant progress
- **Comment Activity**: Summarized review discussions, feedback, and collaborative decisions

### Digest Structure

- **Period at a Glance**: Brief overview of key developments and statistics
- **Development & Delivery**: Summary of merge requests and delivered features
- **Issues & Bug Fixes**: Overview of reported issues and completed fixes
- **Project Planning**: Progress on initiatives and milestones
- **Infrastructure & Operations**: Pipeline health and operational updates
- **Collaboration Highlights**: Notable discussions, decisions, and team interactions
- **Looking Ahead**: Implications for future work based on current activity

### Enhanced Features

- **Clickable Links**: All issue (#123) and merge request (!456) references include direct links
- **Comment Summaries**: Conversations are summarized holistically rather than comment-by-comment
- **Activity Labels**: Applied labels include "activity-digest", "documentation", and "automated"
- **Structured Format**: Clean, scannable layout optimized for quick review

## How to Use

**Note:** This workflow is intended for remote execution and will automatically create a summary issue in your project.

### Configuration

The workflow requires a date range parameter in the format `YYYY-MM-DD-YYYY-MM-DD` (e.g., `2024-01-15-2024-01-21` for a weekly summary).

### Execution Steps

1. Navigate to your GitLab project where you want to generate the activity summary.
1. Look for the "Generate Project Activity Summary with Duo" option in the project interface.
1. Specify your desired date range when prompted (format: `YYYY-MM-DD-YYYY-MM-DD`).
1. Click the button to start the workflow - you should see a flash message "Workflow started successfully."
1. Go to Build > Pipelines from the side menu bar. Check the most recent pipeline with a `workload` job for execution logs.
1. Once the pipeline has successfully executed, go to Issues from the side menu bar. You should see a new issue titled "Activity Digest - [your date range]".
1. Review the generated summary issue for insights and project activity details.
