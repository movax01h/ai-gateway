import os
import sys

from google.api_core.exceptions import AlreadyExists
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud.discoveryengine_v1.types import IndustryVertical, SolutionType

# pylint: disable=direct-environment-variable-reference
if os.environ.get("INGEST_DRY_RUN") == "true":
    print("INFO: Dry Run mode. Skipped.")
    sys.exit(0)

project_id = os.environ["GCP_PROJECT_NAME"]
data_store_id = os.environ["DATA_STORE_ID"]
bigquery_dataset = os.environ["BIGQUERY_DATASET_ID"]
bigquery_table = os.environ["BIGQUERY_TABLE_NAME"]
# pylint: enable=direct-environment-variable-reference


def create_data_store():
    client = discoveryengine.DataStoreServiceClient()

    # https://cloud.google.com/generative-ai-app-builder/docs/provide-schema
    with open("scripts/ingest/gitlab-docs/steps/schema.json", "r") as f:
        json_schema = f.read()

    schema = discoveryengine.Schema(
        name=f"projects/{project_id}/locations/global/collections/default_collection/dataStores/{data_store_id}/"
        "schemas/default_schema",
        json_schema=json_schema,
    )

    data_store = discoveryengine.DataStore(
        display_name=data_store_id,
        industry_vertical=IndustryVertical.GENERIC,
        solution_types=[SolutionType.SOLUTION_TYPE_SEARCH],
        starting_schema=schema,
    )

    request = discoveryengine.CreateDataStoreRequest(
        parent=f"projects/{project_id}/locations/global/collections/default_collection",
        data_store=data_store,
        data_store_id=data_store_id,
    )

    operation = client.create_data_store(request=request)

    print("Waiting for operation to complete...")

    response = operation.result()

    print(response)


def import_documents():
    client = discoveryengine.DocumentServiceClient()

    request = discoveryengine.ImportDocumentsRequest(
        parent=f"projects/{project_id}/locations/global/dataStores/{data_store_id}/branches/default_branch",
        bigquery_source=discoveryengine.BigQuerySource(
            project_id=project_id,
            dataset_id=bigquery_dataset,
            table_id=bigquery_table,
            data_schema="custom",
        ),
        reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.FULL,
        auto_generate_ids=True,
    )

    operation = client.import_documents(request=request)

    print(f"Waiting for operation to complete: {operation.operation.name}")
    response = operation.result()

    metadata = discoveryengine.ImportDocumentsMetadata(operation.metadata)

    print(response)
    print(metadata)


if __name__ == "__main__":
    try:
        create_data_store()
    except AlreadyExists as ex:
        print(f"INFO: Data store already exists. Skipped. {ex}")

    import_documents()
