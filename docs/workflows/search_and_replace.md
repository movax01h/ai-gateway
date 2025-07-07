# Search and Replace Flow

## What It Does

Intelligently updates code across multiple files using pattern-based rules while maintaining code structure and
functionality.

## Prerequisites

Create a configuration file: `.duo_workflow/search_and_replace_config.yaml`

## Configuration Format

```YAML
domain_speciality: "Frontend Development"
assignment_description: "Add accessibility attributes to UI components"
file_types:
  - "*.tsx"
  - "*.jsx"
replacement_rules:
  - element: "button"
    rules: "Add aria-label and ensure keyboard navigation"
  - element: "img"
    rules: "Add descriptive alt text based on context"
```

## Common Use Cases

### Accessibility Compliance

```YAML
domain_speciality: "React Frontend"
assignment_description: "Implement WCAG 2.1 AA standards"
file_types: [ "*.jsx", "*.tsx" ]
replacement_rules:
  - element: "form inputs"
    rules: "Add labels, aria-describedby for errors, and required indicators"
```

### API Migration

```YAML
domain_speciality: "Backend Services"
assignment_description: "Update API from v1 to v2"
file_types: [ "*.js", "*.ts" ]
replacement_rules:
  - element: "API endpoints"
    rules: "Change /api/v1/ to /api/v2/ and update payload structure"
```

### Security Updates

```YAML
domain_speciality: "Full Stack"
assignment_description: "Fix XSS vulnerabilities"
file_types: [ "*.jsx", "*.ejs" ]
replacement_rules:
  - element: "user input rendering"
    rules: "Add proper sanitization before displaying user content"
```

## How to Use

1. Create a .duo_workflow directory in target repository's root folder. Create pipeline config YAML
   search_and_replace_config file in the `.duo_workflow` directory. A sample config should look like

``` plaintext
domain_speciality: "an experienced Vue.js engineer specialised in UI accessibility" # short description of role / position that model should roleplay into
assignment_description: "accessibility issues" # few words description of assignment type, eg: a11y issues, fearture flag removel, eslint error fixes
file_types:
  - "*.vue"
replacement_rules:
  - element: gl-icon # this is going to be an anchor to filter out irrelevant files using grep, in detail analysis and further decision making is going to be done by model according to rules
    rules: |-
      1. Verify that 'aria-label' attribute is being present in each gl-icon dom element
      3. If 'aria-label' attribute is missing from gl-icon element, check if any neighbour element is a text element that duplicates information which could go into 'aria-label' if so do not modify `gl-icon` element.
      4. If you do not know what value 'aria-label' should have add `aria-label` attribute to gl-icon element with value '<placeholder FIX ME>'
  - element: gl-avatar
    rules: |-
      1. Verify that 'alt' attribute is being present in each gl-avatar dom element
      3. If 'alt' attribute is missing from gl-avatar element, check if any neighbour element is a text element that duplicates information which could go into 'alt' if so add 'alt' attribute with empty string.
      4. If you do not know what value 'alt' should have add `alt` attribute to the gl-avatar with value '<placeholder FIX ME>'
```

1. Point goal to a directory (using relative path from repository root) that you wish to process eg export DW_GOAL="
   app/assets/javascripts/admin/abuse_report"
1. Run the executor process locally (or with Docker) following instructions
   in [Running executor locally](https://gitlab.com/gitlab-org/duo-workflow/duo-workflow-executor#running-the-executor).
   Be sure to change the
   `workflow_definition` to `search_and_replace` (--definition="search_and_replace") and the goal to `DW_GOAL` from the
   previous step (--goal="app/assets/javascripts/admin/abuse_report").
1. Verify changes made in the desired directory.

## Best Practices

### Do

- Write specific, actionable rules
- Test patterns from `element` attributes on small subsets first
- Use version control
- Be explicit about edge cases

### Don't

- Use vague rules like "fix all issues"
- Run on entire codebase without testing
- Skip the review step

## Troubleshooting

**No files processed?**

- Check file_types patterns match your files
- Verify config file location

**Unexpected changes?**

- Make rules more specific
- Add exclusion patterns
- Review the execution logs

## Limitations

- Cannot execute code to verify changes
- Large files may hit token limits
- Binary files cannot be processed
