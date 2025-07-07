# Convert to GitLab CI Flow

## What It Does

Automatically converts Jenkins pipelines (Jenkinsfile) to GitLab CI/CD configurations.

## When to Use

- Migrating from Jenkins to GitLab

## Supported Features

### Converts

- Pipeline stages and steps
- Environment variables
- Build triggers and parameters
- Artifacts and dependencies
- Parallel execution
- Conditional logic
- Post-build actions

## Example Conversion

### Jenkins Input

```groovy
pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'npm install'
                sh 'npm build'
            }
        }
        stage('Test') {
            steps {
                sh 'npm test'
            }
        }
        stage('Deploy') {
            when { branch 'main' }
            steps {
                sh './deploy.sh'
            }
        }
    }
}
```

### GitLab Output

```YAML
stages:
  - build
  - test
  - deploy

build:
  stage: build
  script:
    - npm install
    - npm build
  artifacts:
    paths:
      - node_modules/
      - dist/

test:
  stage: test
  script:
    - npm test

deploy:
  stage: deploy
  script:
    - ./deploy.sh
  only:
    - main
```

## How to Use

note: This workflow is intended for remote execution as it pushes the code and creates a merge request automatically.

1. Open the Jenkinsfile in your remote project repository.
1. You should see a button `Convert to GitLab CI/CD` in the file view header section. Click the button, you should see a
   flash message Workflow started successfully.
1. Go to Build > Pipelines from the side menu bar. Checkout the most recent pipeline with a `workload` job for execution
   logs.
1. Once the pipeline has successfully executed, go to Merge Requests from the side menu bar. You should see a merge
   request titled 'Duo Workflow: Convert to GitLab CI'
1. Review the newly created/updated `.gitlab-ci.yml`

## Best Practices

1. **Document dependencies** before migration
1. **List all credentials** that need migration
1. **Run both systems** in parallel initially
1. **Compare outputs** to ensure accuracy
1. **Update documentation** and train team
