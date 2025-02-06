import argparse
import concurrent.futures
import time
from random import choices

from litellm import completion

# List of countries to fill the prompt 'Tell me the history of {country}'
countries = [
    "Afghanistan",
    "Albania",
    "Algeria",
    "Andorra",
    "Angola",
    "Antigua and Barbuda",
    "Argentina",
    "Armenia",
    "Australia",
    "Austria",
    "Azerbaijan",
    "The Bahamas",
    "Bahrain",
    "Bangladesh",
    "Barbados",
    "Belarus",
    "Belgium",
    "Belize",
    "Benin",
    "Bhutan",
    "Bolivia",
    "Bosnia and Herzegovina",
    "Botswana",
    "Brazil",
    "Brunei",
    "Bulgaria",
    "Burkina Faso",
    "Burundi",
    "Cabo Verde",
    "Cambodia",
    "Cameroon",
    "Canada",
    "Central African Republic",
    "Chad",
    "Chile",
    "China",
    "Colombia",
    "Comoros",
    "Congo, Democratic Republic of the",
    "Congo, Republic of the",
    "Costa Rica",
    "Côte d’Ivoire",
    "Croatia",
    "Cuba",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Djibouti",
    "Dominica",
    "Dominican Republic",
    "East Timor (Timor-Leste)",
    "Ecuador",
    "Egypt",
    "El Salvador",
    "Equatorial Guinea",
    "Eritrea",
    "Estonia",
    "Eswatini",
    "Ethiopia",
    "Fiji",
    "Finland",
    "France",
    "Gabon",
    "The Gambia",
    "Georgia",
    "Germany",
    "Ghana",
    "Greece",
    "Grenada",
    "Guatemala",
    "Guinea",
    "Guinea-Bissau",
    "Guyana",
    "Haiti",
    "Honduras",
    "Hungary",
    "Iceland",
    "India",
    "Indonesia",
    "Iran",
    "Iraq",
    "Ireland",
    "Israel",
    "Italy",
    "Jamaica",
    "Japan",
    "Jordan",
    "Kazakhstan",
    "Kenya",
    "Kiribati",
    "Korea, North",
    "Korea, South",
    "Kosovo",
    "Kuwait",
    "Kyrgyzstan",
    "Laos",
    "Latvia",
    "Lebanon",
    "Lesotho",
    "Liberia",
    "Libya",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Madagascar",
    "Malawi",
    "Malaysia",
    "Maldives",
    "Mali",
    "Malta",
    "Marshall Islands",
    "Mauritania",
    "Mauritius",
    "Mexico",
    "Micronesia, Federated States of",
    "Moldova",
    "Monaco",
    "Mongolia",
    "Montenegro",
    "Morocco",
    "Mozambique",
    "Myanmar (Burma)",
    "Namibia",
    "Nauru",
    "Nepal",
    "Netherlands",
    "New Zealand",
    "Nicaragua",
    "Niger",
    "Nigeria",
    "North Macedonia",
    "Norway",
    "Oman",
    "Pakistan",
    "Palau",
    "Panama",
    "Papua New Guinea",
    "Paraguay",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Qatar",
    "Romania",
    "Russia",
    "Rwanda",
    "Saint Kitts and Nevis",
    "Saint Lucia",
    "Saint Vincent and the Grenadines",
    "Samoa",
    "San Marino",
    "Sao Tome and Principe",
    "Saudi Arabia",
    "Senegal",
    "Serbia",
    "Seychelles",
    "Sierra Leone",
    "Singapore",
    "Slovakia",
    "Slovenia",
    "Solomon Islands",
    "Somalia",
    "South Africa",
    "Spain",
    "Sri Lanka",
    "Sudan",
    "Sudan, South",
    "Suriname",
    "Sweden",
    "Switzerland",
    "Syria",
    "Taiwan",
    "Tajikistan",
    "Tanzania",
    "Thailand",
    "Togo",
    "Tonga",
    "Trinidad and Tobago",
    "Tunisia",
    "Turkey",
    "Turkmenistan",
    "Tuvalu",
    "Uganda",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
    "Uruguay",
    "Uzbekistan",
    "Vanuatu",
    "Vatican City",
    "Venezuela",
    "Vietnam",
    "Yemen",
    "Zambia",
]


def make_request(model_config, country_name):
    start = time.time()
    query = f"Tell me the story of {country_name}"
    response = completion(
        model=model_config["model_identifier"],
        messages=[{"content": query, "role": "user"}],
        stream=False,
        api_key=model_config["api_key"],
        base_url=model_config["model_endpoint"],
        max_tokens=2000,
    )

    time_passed = time.time() - start

    return {
        "query": query,
        "time_passed": time_passed,
        "tokens": response.usage.completion_tokens,
        "tokens_per_second": response.usage.completion_tokens / time_passed,
    }


def run_parallel_operations(model_config, num_operations=10):
    start = time.time()

    countries_to_call = choices(countries, k=num_operations)

    # Use ThreadPoolExecutor for I/O-bound tasks or ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_operations) as executor:
        # Submit tasks to the executor
        futures = [
            executor.submit(make_request, model_config, c) for c in countries_to_call
        ]

        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    total_time = time.time() - start
    return {"results": results, "total_time": total_time}


def compute_tps():
    parser = argparse.ArgumentParser(
        description="Compute TPS of a specified model, prints the summary as markdown"
    )
    parser.add_argument(
        "--model-endpoint",
        required=False,
        default=None,
        help="Endpoint of the model. Example: http://localhost:4000. "
        "When using a model from an online provider like Bedrock, "
        "this can be left empty.",
    )
    parser.add_argument(
        "--model-identifier", required=False, help="Identifier of the model"
    )
    parser.add_argument("--api-key", required=False, help="API key for the model.")
    parser.add_argument(
        "--requests",
        required=False,
        nargs="*",
        help="Number of requests to send. Accepts a list of values.",
        default=["1"],
    )
    parser.add_argument(
        "--md",
        required=False,
        help="Print as markdown table",
        action="store_true",
    )

    args = parser.parse_args()

    model_name = args.model_identifier

    model_config = {
        "model_endpoint": args.model_endpoint,
        "api_key": args.api_key,
        "model_identifier": model_name,
    }

    if args.md:
        print(
            "| Model name | Number of requests | Average time per request (s) | Average tokens in response | "
            "Average tokens per second per request |  Total time for requests	| Total TPS |"
        )
        print("| --- | --- | --- | ---	| --- |  --- | --- |")

    for requests_in_batch in args.requests:
        n_requests = int(requests_in_batch)
        results = run_parallel_operations(model_config, int(n_requests))

        total_tokens = sum(i["tokens"] for i in results["results"])
        total_time = results["total_time"]
        total_tps = total_tokens / results["total_time"]

        avg_request_time = (
            sum(i["time_passed"] for i in results["results"]) * 1.0 / n_requests
        )
        avg_tokens = sum(i["tokens"] for i in results["results"]) * 1.0 / n_requests
        avg_tps = (
            sum(i["tokens_per_second"] for i in results["results"]) * 1.0 / n_requests
        )

        if args.md:
            print(
                f"| {model_name} | {n_requests} | {avg_request_time:.2f} |  {avg_tokens} | {avg_tps:.2f} "
                f"|  {total_time:.2f} | {total_tps:.2f} |"
            )
        else:
            print(f"Number of Requests: {requests_in_batch}")
            print(f"Average time per request (s): {avg_request_time:.2f}")
            print(f"Average tokens in response: {avg_tokens}")
            print(f"Average tokens per second per request: {avg_tps:.2f}")
            print(f"Total time for requests (s): {total_time:.2f}")
            print(f"Total TPS: {total_tps:.2f}")
            print("")
