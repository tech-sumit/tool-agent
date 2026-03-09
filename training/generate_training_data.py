"""Generate xLAM-2 training data from n8n knowledge database.

Parses node_knowledge_db.json and connection_patterns.json to produce
JSONL training examples in the xLAM-2 conversation format.

All training prompts and scenarios are loaded from CSV files in
training/data/ — edit those files to add or modify training examples
without touching this script.

Output format per line (JSON):
  {
    "messages": [
      {"role": "system", "content": "...tool schemas..."},
      {"role": "user", "content": "user request"},
      {"role": "assistant", "content": "[{\"name\": ..., \"arguments\": {...}}]"}
    ]
  }

Five categories of training examples are generated:
  1. Tool Selection — pick the right n8n node for a natural-language request
  2. Parameter Filling — select tool AND fill realistic parameter values
  3. Multi-tool Composition — chain nodes for multi-step workflows
  4. Tool Discovery — answer "what tools can do X?" queries
  5. Negative — correctly refuse when no tool is suitable
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

XLAM_SYSTEM_PROMPT = (
    "You are a helpful assistant that can use tools.\n"
    "You have access to a set of tools. When using tools, make calls "
    "in a single JSON array:\n\n"
    '[{"name": "tool_call_name", "arguments": {"arg1": "value1", '
    '"arg2": "value2"}}, ... (additional parallel tool calls as needed)]\n\n'
    "If no tool is suitable, state that explicitly. If the user's input "
    "lacks required parameters, ask for clarification. Do not interpret "
    "or respond until tool results are returned. Once they are available, "
    "process them or make additional calls if needed. For tasks that don't "
    "require tools, such as casual conversation or general advice, respond "
    "directly in plain text. The available tools are:"
)

NODE_PREFIX_MAP = {
    "n8n-nodes-base.": "",
    "@n8n/n8n-nodes-langchain.": "ai_",
    "@blotato/n8n-nodes-blotato.": "blotato_",
}

DATA_DIR = Path(__file__).parent / "data"


# ── CSV loaders ──────────────────────────────────────────────────────


def load_integration_descriptions(path: Path | None = None) -> dict[str, str]:
    path = path or DATA_DIR / "integration_descriptions.csv"
    descriptions: dict[str, str] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            descriptions[row["node_name"].strip()] = row["description"].strip()
    return descriptions


def load_task_templates(path: Path | None = None) -> list[str]:
    path = path or DATA_DIR / "task_templates.csv"
    templates: list[str] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            templates.append(row["template"].strip())
    return templates


def load_single_tool_tasks(path: Path | None = None) -> dict[str, list[str]]:
    path = path or DATA_DIR / "single_tool_tasks.csv"
    tasks: dict[str, list[str]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tasks[row["node_type"].strip()].append(row["task"].strip())
    return dict(tasks)


def load_multi_step_scenarios(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or DATA_DIR / "multi_step_scenarios.csv"
    scenarios: list[dict[str, Any]] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            scenarios.append({
                "prompt": row["prompt"].strip(),
                "tools": [t.strip() for t in row["tools"].split(",")],
            })
    return scenarios


def load_discovery_prompts(path: Path | None = None) -> dict[str, list[str]]:
    path = path or DATA_DIR / "discovery_prompts.csv"
    prompts: dict[str, list[str]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            prompts[row["category"].strip()].append(row["prompt"].strip())
    return dict(prompts)


def load_negative_examples(path: Path | None = None) -> list[str]:
    path = path or DATA_DIR / "negative_examples.csv"
    prompts: list[str] = []
    if not path.exists():
        return prompts
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            prompts.append(row["prompt"].strip())
    return prompts


# ── Schema building ─────────────────────────────────────────────────


def clean_node_type(raw_type: str) -> str:
    for prefix, replacement in NODE_PREFIX_MAP.items():
        if raw_type.startswith(prefix):
            return replacement + raw_type[len(prefix):]
    parts = raw_type.rsplit(".", 1)
    return parts[-1] if len(parts) > 1 else raw_type


def build_function_schema(
    node_name: str,
    node_data: dict,
    descriptions: dict[str, str],
) -> dict:
    """Build a JSON-Schema-style function descriptor from node_knowledge_db entry."""
    clean_name = clean_node_type(node_name)
    description = descriptions.get(
        clean_name,
        f"n8n integration: {clean_name} (used {node_data.get('total_uses', 0)} times)",
    )

    properties: dict[str, dict] = {}
    required: list[str] = []

    type_map = {
        "str": "string", "int": "integer", "float": "number",
        "bool": "boolean", "dict": "object", "list": "array",
    }

    for param_key, param_data in node_data.get("parameters", {}).items():
        if "." in param_key or param_key == "options":
            continue

        usage = param_data.get("usage_count", 0)
        if usage < 10:
            continue

        value_types = param_data.get("value_types", {})
        samples = param_data.get("sample_values", [])

        json_type = "string"
        if isinstance(value_types, dict) and value_types:
            dominant_type = max(value_types, key=value_types.get)
            json_type = type_map.get(dominant_type, "string")
        elif isinstance(value_types, list) and value_types:
            first = value_types[0]
            if isinstance(first, str):
                json_type = type_map.get(first, "string")

        prop: dict[str, Any] = {"type": json_type}
        if samples:
            clean_samples = [
                s for s in samples[:3]
                if isinstance(s, str) and len(s) < 80 and not s.startswith("=")
            ]
            if clean_samples:
                prop["examples"] = clean_samples

        prop["description"] = f"{param_key.replace('_', ' ')} parameter"
        properties[param_key] = prop

        ratio = usage / max(node_data.get("total_uses", 1), 1)
        if ratio > 0.6:
            required.append(param_key)

    if not properties:
        properties = {"input": {"type": "string", "description": "Input data"}}

    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": clean_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        },
    }
    if required:
        schema["function"]["parameters"]["required"] = required

    return schema


def format_tools_for_system(tools: list[dict]) -> str:
    """Format tool schemas for the system message."""
    parts = []
    for tool in tools:
        func = tool.get("function", tool)
        parts.append(json.dumps(func))
    return "\n\n".join(parts)


def build_system_message(tools: list[dict]) -> str:
    return f"{XLAM_SYSTEM_PROMPT}\n\n{format_tools_for_system(tools)}"


def make_sample_args(schema: dict) -> dict[str, Any]:
    """Generate realistic sample argument values from a tool schema."""
    func = schema.get("function", schema)
    params = func.get("parameters", {})
    properties = params.get("properties", {})
    required = params.get("required", [])

    args: dict[str, Any] = {}
    for key in required:
        prop = properties.get(key, {})
        ptype = prop.get("type", "string")
        examples = prop.get("examples", [])
        if examples:
            args[key] = random.choice(examples)
        elif ptype == "string":
            args[key] = f"sample_{key}"
        elif ptype == "integer":
            args[key] = random.randint(1, 100)
        elif ptype == "number":
            args[key] = round(random.uniform(0, 100), 2)
        elif ptype == "boolean":
            args[key] = random.choice([True, False])
        elif ptype == "object":
            args[key] = {}
        elif ptype == "array":
            args[key] = []
    return args


# ── Example generators ───────────────────────────────────────────────


def generate_tool_selection_examples(
    schemas: dict[str, dict],
    single_tool_tasks: dict[str, list[str]],
    task_templates: list[str],
) -> list[dict]:
    """Generate single-tool selection training examples in xLAM-2 format."""
    examples = []

    for node_type, tasks in single_tool_tasks.items():
        clean_name = node_type
        matching_schemas = [
            s for s in schemas.values() if s["function"]["name"] == clean_name
        ]
        if not matching_schemas:
            continue

        target_schema = matching_schemas[0]
        distractor_keys = [
            k for k in schemas if schemas[k]["function"]["name"] != clean_name
        ]

        for task_text in tasks:
            random.shuffle(distractor_keys)
            n_distractors = random.randint(3, 8)
            distractors = [schemas[k] for k in distractor_keys[:n_distractors]]
            available_tools = [target_schema] + distractors
            random.shuffle(available_tools)

            n_templates = min(3, len(task_templates))
            for template in random.sample(task_templates, n_templates):
                if "{action_s}" in template:
                    action = task_text + "s" if not task_text.endswith("s") else task_text
                    prompt = template.replace("{action_s}", action)
                else:
                    prompt = template.replace("{action}", task_text)

                sample_args = make_sample_args(target_schema)
                tool_call = [{"name": clean_name, "arguments": sample_args}]

                examples.append({
                    "messages": [
                        {"role": "system", "content": build_system_message(available_tools)},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": json.dumps(tool_call)},
                    ],
                    "category": "tool_selection",
                })

    return examples


def generate_param_filling_examples(
    schemas: dict[str, dict],
) -> list[dict]:
    """Generate examples with fully filled realistic parameters."""
    examples = []

    param_prompts = {
        "slack": [
            ("Send a message to #engineering saying 'Deploy complete'",
             {"channel": "#engineering", "text": "Deploy complete"}),
            ("Post 'Build failed' to the #alerts Slack channel",
             {"channel": "#alerts", "text": "Build failed"}),
            ("DM @john with 'Meeting moved to 3pm'",
             {"channel": "@john", "text": "Meeting moved to 3pm"}),
        ],
        "gmail": [
            ("Send an email to alice@example.com with subject 'Report Ready'",
             {"to": "alice@example.com", "subject": "Report Ready"}),
            ("Email bob@corp.com about the quarterly review",
             {"to": "bob@corp.com", "subject": "quarterly review"}),
        ],
        "telegram": [
            ("Send 'Hello World' to chat ID 12345",
             {"chatId": "12345", "text": "Hello World"}),
            ("Send a notification to Telegram chat 98765",
             {"chatId": "98765", "text": "notification"}),
        ],
        "postgres": [
            ("Query all users from the users table",
             {"query": "SELECT * FROM users"}),
            ("Insert a new row into orders table",
             {"query": "INSERT INTO orders (product, qty) VALUES ('Widget', 10)"}),
            ("Count active customers in the database",
             {"query": "SELECT COUNT(*) FROM customers WHERE active = true"}),
        ],
        "httpRequest": [
            ("GET the list of users from https://api.example.com/users",
             {"url": "https://api.example.com/users", "method": "GET"}),
            ("POST data to https://hooks.example.com/webhook",
             {"url": "https://hooks.example.com/webhook", "method": "POST"}),
        ],
        "googleSheets": [
            ("Read all rows from the 'Sales' sheet",
             {"operation": "read", "sheetName": "Sales"}),
            ("Append a row to the 'Leads' spreadsheet",
             {"operation": "append", "sheetName": "Leads"}),
        ],
        "notion": [
            ("Create a page titled 'Meeting Notes' in Notion",
             {"title": "Meeting Notes"}),
            ("Search for pages about 'project plan' in Notion",
             {"query": "project plan"}),
        ],
        "github": [
            ("Create an issue titled 'Fix login bug' in repo myorg/myapp",
             {"title": "Fix login bug", "owner": "myorg", "repo": "myapp"}),
            ("List open PRs on myorg/backend",
             {"owner": "myorg", "repo": "backend"}),
        ],
        "openAi": [
            ("Summarize this text using GPT-4",
             {"model": "gpt-4", "prompt": "Summarize the following text"}),
            ("Generate an image of a sunset over mountains",
             {"prompt": "sunset over mountains"}),
        ],
        "discord": [
            ("Send 'Server maintenance at 10pm' to the #announcements channel",
             {"channel": "#announcements", "content": "Server maintenance at 10pm"}),
        ],
    }

    for node_type, prompt_pairs in param_prompts.items():
        matching = [s for s in schemas.values() if s["function"]["name"] == node_type]
        if not matching:
            continue

        target_schema = matching[0]
        distractor_keys = [
            k for k in schemas if schemas[k]["function"]["name"] != node_type
        ]

        for prompt_text, expected_args in prompt_pairs:
            random.shuffle(distractor_keys)
            distractors = [schemas[k] for k in distractor_keys[:random.randint(4, 8)]]
            available_tools = [target_schema] + distractors
            random.shuffle(available_tools)

            tool_call = [{"name": node_type, "arguments": expected_args}]

            examples.append({
                "messages": [
                    {"role": "system", "content": build_system_message(available_tools)},
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": json.dumps(tool_call)},
                ],
                "category": "param_filling",
            })

    return examples


def generate_composition_examples(
    schemas: dict[str, dict],
    scenarios: list[dict[str, Any]],
) -> list[dict]:
    """Generate multi-tool composition training examples."""
    examples = []

    for scenario in scenarios:
        tool_names = scenario["tools"]
        tool_schemas = []
        for name in tool_names:
            matching = [s for s in schemas.values() if s["function"]["name"] == name]
            if matching:
                tool_schemas.append(matching[0])

        if len(tool_schemas) < 2:
            continue

        distractor_keys = [
            k for k in schemas
            if schemas[k]["function"]["name"] not in tool_names
        ]
        random.shuffle(distractor_keys)
        distractors = [schemas[k] for k in distractor_keys[:random.randint(3, 6)]]
        available_tools = tool_schemas + distractors
        random.shuffle(available_tools)

        tool_calls = [
            {"name": name, "arguments": make_sample_args(
                next((s for s in schemas.values() if s["function"]["name"] == name), {})
            )}
            for name in tool_names
            if any(s["function"]["name"] == name for s in schemas.values())
        ]

        if not tool_calls:
            continue

        examples.append({
            "messages": [
                {"role": "system", "content": build_system_message(available_tools)},
                {"role": "user", "content": scenario["prompt"]},
                {"role": "assistant", "content": json.dumps(tool_calls)},
            ],
            "category": "composition",
        })

    return examples


def generate_discovery_examples(
    schemas: dict[str, dict],
    discovery_prompts: dict[str, list[str]],
) -> list[dict]:
    """Generate tool discovery/recommendation training examples."""
    examples = []

    categories: dict[str, list[dict]] = defaultdict(list)
    for schema in schemas.values():
        desc = schema["function"]["description"].lower()
        for keyword in discovery_prompts:
            if keyword in desc:
                categories[keyword].append(schema)

    all_tools = list(schemas.values())[:30]

    for cat, prompts in discovery_prompts.items():
        if cat not in categories:
            continue
        cat_tools = categories[cat]
        tool_list = ", ".join(t["function"]["name"] for t in cat_tools[:10])

        for prompt in prompts:
            response = (
                f"Available tools for {cat}: {tool_list}. "
                "Let me know which one you'd like to use and I can help set it up."
            )

            examples.append({
                "messages": [
                    {"role": "system", "content": build_system_message(all_tools)},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "category": "discovery",
            })

    return examples


def generate_negative_examples(
    schemas: dict[str, dict],
    negative_prompts: list[str],
) -> list[dict]:
    """Generate examples where no tool should be called."""
    examples = []
    all_tools = list(schemas.values())[:20]

    refusals = [
        "None of the available tools are suitable for this request. "
        "The tools I have are for workflow automation, API calls, "
        "and data integration — not general knowledge questions.",
        "I don't have a tool for that. My tools are focused on "
        "n8n workflow operations like sending messages, querying databases, "
        "and calling APIs. This is a general knowledge question.",
        "This doesn't require any of my available tools. I can help with "
        "tasks like sending emails, posting to Slack, querying databases, "
        "or calling APIs. Let me know if you need help with something like that.",
    ]

    for prompt in negative_prompts:
        random.shuffle(all_tools)
        subset = all_tools[:random.randint(5, 15)]

        examples.append({
            "messages": [
                {"role": "system", "content": build_system_message(subset)},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": random.choice(refusals)},
            ],
            "category": "negative",
        })

    return examples


# ── Main ─────────────────────────────────────────────────────────────


def load_and_process(
    knowledge_db_path: Path,
    connections_path: Path,
    descriptions: dict[str, str],
) -> tuple[dict[str, dict], dict]:
    with open(knowledge_db_path) as f:
        node_db = json.load(f)

    with open(connections_path) as f:
        connection_patterns = json.load(f)

    schemas: dict[str, dict] = {}
    for node_name, node_data in node_db.items():
        if node_data.get("total_uses", 0) < 5:
            continue
        schema = build_function_schema(node_name, node_data, descriptions)
        schemas[node_name] = schema

    return schemas, connection_patterns


def main():
    parser = argparse.ArgumentParser(description="Generate xLAM-2 training data")
    parser.add_argument("--knowledge-db", required=True, type=Path)
    parser.add_argument("--connections", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = args.data_dir

    print(f"Loading CSV data from {data_dir}/...")
    descriptions = load_integration_descriptions(data_dir / "integration_descriptions.csv")
    task_templates = load_task_templates(data_dir / "task_templates.csv")
    single_tool_tasks = load_single_tool_tasks(data_dir / "single_tool_tasks.csv")
    scenarios = load_multi_step_scenarios(data_dir / "multi_step_scenarios.csv")
    discovery_prompts = load_discovery_prompts(data_dir / "discovery_prompts.csv")
    negative_prompts = load_negative_examples(data_dir / "negative_examples.csv")

    print(f"  {len(descriptions)} integration descriptions")
    print(f"  {len(task_templates)} task templates")
    print(f"  {sum(len(v) for v in single_tool_tasks.values())} single-tool tasks across {len(single_tool_tasks)} node types")
    print(f"  {len(scenarios)} multi-step scenarios")
    print(f"  {sum(len(v) for v in discovery_prompts.values())} discovery prompts across {len(discovery_prompts)} categories")
    print(f"  {len(negative_prompts)} negative examples")

    print(f"\nLoading knowledge DB from {args.knowledge_db}...")
    schemas, connection_patterns = load_and_process(
        args.knowledge_db, args.connections, descriptions,
    )
    print(f"  Built {len(schemas)} function schemas")

    print("\nGenerating examples...")
    selection = generate_tool_selection_examples(schemas, single_tool_tasks, task_templates)
    print(f"  Tool selection:  {len(selection)}")

    params = generate_param_filling_examples(schemas)
    print(f"  Param filling:   {len(params)}")

    composition = generate_composition_examples(schemas, scenarios)
    print(f"  Composition:     {len(composition)}")

    discovery = generate_discovery_examples(schemas, discovery_prompts)
    print(f"  Discovery:       {len(discovery)}")

    negative = generate_negative_examples(schemas, negative_prompts)
    print(f"  Negative:        {len(negative)}")

    all_examples = selection + params + composition + discovery + negative
    random.shuffle(all_examples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    cats = defaultdict(int)
    for ex in all_examples:
        cats[ex.get("category", "unknown")] += 1
    print(f"\nTotal: {len(all_examples)} examples written to {args.output}")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
