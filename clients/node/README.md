# GitLab Duo Workflow Service Node Client

A TypeScript/JavaScript client for the GitLab Duo Workflow Service.

## Installation

```bash
npm install @gitlab-org/duo-workflow-service
```

Note that this package is available on the GitLab package registry, not on `npm`

## Usage

```typescript
import {
  Action,
  DuoWorkflowClient,
  ChannelCredentials,
  Metadata,
} from "@gitlab-org/duo-workflow-service";

const duoWorkflowServiceBaseUrl = "localhost:50051";
const credentials = ChannelCredentials.createInsecure(); // Or ChannelCredentials.createSsl
const client = new DuoWorkflowClient(duoWorkflowServiceBaseUrl, credentials);

const metadata: Metadata = {
  // See type definition
};
const stream = client.executeWorkflow(metadata);

stream.on("data", async (action: Action) => {
  console.debug("Got action:", action);
});
stream.on("error", (error) => console.error("oh no! ", error));
stream.on("end", () => console.info("Stream ended"));
```

## Development

### Building the client

The client is generated from the Duo Workflow Service build.

```bash
make gen-proto-node
```

### Publishing

You should have generated, built, and incremented the version locally, and committed your changes.

Then when your change hits CI for `main`, a `publish-node-client` job will run. If the existing published version is not the same as the new version, the new version will be published to the GitLab Package Registry.

#### Versioning

There is a simple script that automatically increments the package.json version when running `make gen-proto-node`, if the generated file changes.

The script does not use proper SemVer, it will always increment the `patch` version each time there is any change. Feel free to manually update the `minor` or `major` if your change is significant / breaking.

#### Testing before publishing

If you want to test the new client version, say, in the GitLab Language Server project, _before_ you have published, you can use `yalc` to link the local build.

1. Generate and build the client with `make gen-proto-node`.
1. `cd clients/node`
1. `npx yalc publish --push`

Your development version is now pushed to the local yalc store and can be installed into other projects.
If you make changes, re-run the yalc publish command.

1. In the Language Server, cd to the package which uses this client, `cd packages/lib_workflow_executor`
1. `npx yalc add @gitlab-org/duo-workflow-service` to install your local version of the package
1. Re-run the Language Server build to pick up the new package

When you are done, remove the yalc changes via `git` or `npx yalc remove @gitlab-org/duo-workflow-service`.
