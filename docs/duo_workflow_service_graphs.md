
# Duo Workflow Service Graphs

These diagrams show the LangGraph structure of each Workflow in the duo_workflow_service. Do not manually edit
this file, instead update it by running `make duo-workflow-docs`.

## Graph: `software_development`

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
    __start__([<p>__start__</p>]):::first
    build_context(build_context)
    build_context_tools(build_context_tools)
    build_context_handover(build_context_handover)
    build_context_supervisor(build_context_supervisor)
    tools_approval_entry_context_builder(tools_approval_entry_context_builder)
    tools_approval_check_context_builder(tools_approval_check_context_builder)
    task_clarity_build_prompt(task_clarity_build_prompt)
    task_clarity_check(task_clarity_check)
    task_clarity_request_clarification(task_clarity_request_clarification)
    task_clarity_fetch_user_response(task_clarity_fetch_user_response)
    task_clarity_cancel_pending_tool_call(task_clarity_cancel_pending_tool_call)
    task_clarity_handover(task_clarity_handover)
    planning(planning)
    update_plan(update_plan)
    planning_supervisor(planning_supervisor)
    plan_approval_entry_planner(plan_approval_entry_planner)
    plan_approval_check_planner(plan_approval_check_planner)
    plan_terminator(plan_terminator)
    set_status_to_execution(set_status_to_execution)
    execution(execution)
    execution_tools(execution_tools)
    execution_supervisor(execution_supervisor)
    execution_handover(execution_handover)
    tools_approval_entry_executor(tools_approval_entry_executor)
    tools_approval_check_executor(tools_approval_check_executor)
    __end__([<p>__end__</p>]):::last
    __start__ --> build_context;
    build_context_handover --> task_clarity_build_prompt;
    build_context_supervisor --> build_context;
    execution_handover --> __end__;
    execution_supervisor --> execution;
    execution_tools --> execution;
    plan_terminator --> __end__;
    planning_supervisor --> planning;
    set_status_to_execution --> execution;
    task_clarity_build_prompt --> task_clarity_check;
    task_clarity_cancel_pending_tool_call --> task_clarity_handover;
    task_clarity_handover --> planning;
    task_clarity_request_clarification --> task_clarity_fetch_user_response;
    update_plan --> planning;
    tools_approval_entry_context_builder -. &nbsp;continue&nbsp; .-> tools_approval_check_context_builder;
    tools_approval_entry_context_builder -. &nbsp;back&nbsp; .-> build_context;
    tools_approval_check_context_builder -. &nbsp;continue&nbsp; .-> build_context_tools;
    tools_approval_check_context_builder -. &nbsp;back&nbsp; .-> build_context;
    tools_approval_check_context_builder -. &nbsp;stop&nbsp; .-> plan_terminator;
    build_context -. &nbsp;call_tool&nbsp; .-> build_context_tools;
    build_context -. &nbsp;tools_approval&nbsp; .-> tools_approval_entry_context_builder;
    build_context -. &nbsp;HandoverAgent&nbsp; .-> build_context_handover;
    build_context -. &nbsp;PlanSupervisorAgent&nbsp; .-> build_context_supervisor;
    build_context -. &nbsp;stop&nbsp; .-> plan_terminator;
    build_context_tools -.-> build_context;
    build_context_tools -. &nbsp;stop&nbsp; .-> plan_terminator;
    task_clarity_check -. &nbsp;clear&nbsp; .-> task_clarity_handover;
    task_clarity_check -. &nbsp;skip_further_clarification&nbsp; .-> task_clarity_cancel_pending_tool_call;
    task_clarity_check -. &nbsp;unclear&nbsp; .-> task_clarity_request_clarification;
    task_clarity_check -. &nbsp;stop&nbsp; .-> plan_terminator;
    task_clarity_fetch_user_response -. &nbsp;continue&nbsp; .-> task_clarity_check;
    task_clarity_fetch_user_response -. &nbsp;stop&nbsp; .-> plan_terminator;
    plan_approval_entry_planner -. &nbsp;continue&nbsp; .-> plan_approval_check_planner;
    plan_approval_entry_planner -. &nbsp;back&nbsp; .-> planning;
    plan_approval_check_planner -. &nbsp;continue&nbsp; .-> set_status_to_execution;
    plan_approval_check_planner -. &nbsp;back&nbsp; .-> planning;
    plan_approval_check_planner -. &nbsp;stop&nbsp; .-> plan_terminator;
    planning -. &nbsp;call_tool&nbsp; .-> update_plan;
    planning -. &nbsp;PlanSupervisorAgent&nbsp; .-> planning_supervisor;
    planning -. &nbsp;HandoverAgent&nbsp; .-> plan_approval_entry_planner;
    planning -. &nbsp;stop&nbsp; .-> plan_terminator;
    tools_approval_entry_executor -. &nbsp;continue&nbsp; .-> tools_approval_check_executor;
    tools_approval_entry_executor -. &nbsp;back&nbsp; .-> execution;
    tools_approval_check_executor -. &nbsp;continue&nbsp; .-> execution_tools;
    tools_approval_check_executor -. &nbsp;back&nbsp; .-> execution;
    tools_approval_check_executor -. &nbsp;stop&nbsp; .-> plan_terminator;
    execution -. &nbsp;tools_approval&nbsp; .-> tools_approval_entry_executor;
    execution -. &nbsp;call_tool&nbsp; .-> execution_tools;
    execution -. &nbsp;HandoverAgent&nbsp; .-> execution_handover;
    execution -. &nbsp;PlanSupervisorAgent&nbsp; .-> execution_supervisor;
    execution -. &nbsp;stop&nbsp; .-> plan_terminator;
    task_clarity_fetch_user_response -. &nbsp;back&nbsp; .-> task_clarity_fetch_user_response;
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
```

## Graph: `search_and_replace`

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
    __start__([<p>__start__</p>]):::first
    load_config(load_config)
    scan_directory_tree(scan_directory_tree)
    detect_affected_components(detect_affected_components)
    append_affected_file(append_affected_file)
    request_patch(request_patch)
    log_agent_response(log_agent_response)
    apply_patch(apply_patch)
    complete(complete)
    __end__([<p>__end__</p>]):::last
    __start__ --> load_config;
    complete --> __end__;
    load_config --> scan_directory_tree;
    request_patch --> log_agent_response;
    scan_directory_tree -. &nbsp;end&nbsp; .-> complete;
    scan_directory_tree -. &nbsp;continue&nbsp; .-> detect_affected_components;
    detect_affected_components -. &nbsp;continue&nbsp; .-> append_affected_file;
    detect_affected_components -. &nbsp;end&nbsp; .-> complete;
    append_affected_file -. &nbsp;skip&nbsp; .-> detect_affected_components;
    append_affected_file -. &nbsp;continue&nbsp; .-> request_patch;
    append_affected_file -. &nbsp;end&nbsp; .-> complete;
    log_agent_response -. &nbsp;skip&nbsp; .-> detect_affected_components;
    log_agent_response -. &nbsp;continue&nbsp; .-> apply_patch;
    log_agent_response -. &nbsp;end&nbsp; .-> complete;
    apply_patch -. &nbsp;continue&nbsp; .-> detect_affected_components;
    apply_patch -. &nbsp;end&nbsp; .-> complete;
    detect_affected_components -. &nbsp;skip&nbsp; .-> detect_affected_components;
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
```

## Graph: `convert_to_gitlab_ci`

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
    __start__([<p>__start__</p>]):::first
    load_files(load_files)
    request_translation(request_translation)
    execution_tools(execution_tools)
    git_actions(git_actions)
    complete(complete)
    __end__([<p>__end__</p>]):::last
    __start__ --> load_files;
    complete --> __end__;
    git_actions --> complete;
    load_files --> request_translation;
    request_translation -. &nbsp;continue&nbsp; .-> execution_tools;
    request_translation -. &nbsp;end&nbsp; .-> complete;
    execution_tools -. &nbsp;agent&nbsp; .-> request_translation;
    execution_tools -. &nbsp;end&nbsp; .-> complete;
    execution_tools -. &nbsp;commit_changes&nbsp; .-> git_actions;
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
```

## Graph: `chat`

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
    __start__([<p>__start__</p>]):::first
    agent(agent)
    run_tools(run_tools)
    __end__([<p>__end__</p>]):::last
    __start__ --> agent;
    run_tools --> agent;
    agent -. &nbsp;tool_use&nbsp; .-> run_tools;
    agent -. &nbsp;stop&nbsp; .-> __end__;
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
```
