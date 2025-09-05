
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
    planning(planning)
    update_plan(update_plan)
    planning_supervisor(planning_supervisor)
    plan_terminator(plan_terminator)
    set_status_to_execution(set_status_to_execution)
    execution(execution)
    tools_approval_entry_executor(tools_approval_entry_executor)
    tools_approval_check_executor(tools_approval_check_executor)
    execution_tools(execution_tools)
    execution_supervisor(execution_supervisor)
    execution_handover(execution_handover)
    __end__([<p>__end__</p>]):::last
    __start__ --> build_context;
    build_context -. &nbsp;HandoverAgent&nbsp; .-> build_context_handover;
    build_context -. &nbsp;PlanSupervisorAgent&nbsp; .-> build_context_supervisor;
    build_context -. &nbsp;call_tool&nbsp; .-> build_context_tools;
    build_context -. &nbsp;stop&nbsp; .-> plan_terminator;
    build_context -. &nbsp;tools_approval&nbsp; .-> tools_approval_entry_context_builder;
    build_context_handover --> planning;
    build_context_supervisor --> build_context;
    build_context_tools -.-> build_context;
    build_context_tools -. &nbsp;stop&nbsp; .-> plan_terminator;
    execution -. &nbsp;HandoverAgent&nbsp; .-> execution_handover;
    execution -. &nbsp;PlanSupervisorAgent&nbsp; .-> execution_supervisor;
    execution -. &nbsp;call_tool&nbsp; .-> execution_tools;
    execution -. &nbsp;stop&nbsp; .-> plan_terminator;
    execution -. &nbsp;tools_approval&nbsp; .-> tools_approval_entry_executor;
    execution_supervisor --> execution;
    execution_tools --> execution;
    planning -. &nbsp;stop&nbsp; .-> plan_terminator;
    planning -. &nbsp;PlanSupervisorAgent&nbsp; .-> planning_supervisor;
    planning -. &nbsp;HandoverAgent&nbsp; .-> set_status_to_execution;
    planning -. &nbsp;call_tool&nbsp; .-> update_plan;
    planning_supervisor --> planning;
    set_status_to_execution --> execution;
    tools_approval_check_context_builder -. &nbsp;back&nbsp; .-> build_context;
    tools_approval_check_context_builder -. &nbsp;continue&nbsp; .-> build_context_tools;
    tools_approval_check_context_builder -. &nbsp;stop&nbsp; .-> plan_terminator;
    tools_approval_check_executor -. &nbsp;back&nbsp; .-> execution;
    tools_approval_check_executor -. &nbsp;continue&nbsp; .-> execution_tools;
    tools_approval_check_executor -. &nbsp;stop&nbsp; .-> plan_terminator;
    tools_approval_entry_context_builder -. &nbsp;back&nbsp; .-> build_context;
    tools_approval_entry_context_builder -. &nbsp;continue&nbsp; .-> tools_approval_check_context_builder;
    tools_approval_entry_executor -. &nbsp;back&nbsp; .-> execution;
    tools_approval_entry_executor -. &nbsp;continue&nbsp; .-> tools_approval_check_executor;
    update_plan --> planning;
    execution_handover --> __end__;
    plan_terminator --> __end__;
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
    execution_tools -. &nbsp;end&nbsp; .-> complete;
    execution_tools -. &nbsp;commit_changes&nbsp; .-> git_actions;
    execution_tools -. &nbsp;agent&nbsp; .-> request_translation;
    git_actions --> complete;
    load_files --> request_translation;
    request_translation -. &nbsp;end&nbsp; .-> complete;
    request_translation -. &nbsp;continue&nbsp; .-> execution_tools;
    complete --> __end__;
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
    agent -. &nbsp;stop&nbsp; .-> __end__;
    agent -. &nbsp;tool_use&nbsp; .-> run_tools;
    run_tools --> agent;
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
```

## Graph: `issue_to_merge_request`

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
    planning(planning)
    update_plan(update_plan)
    planning_supervisor(planning_supervisor)
    plan_terminator(plan_terminator)
    set_status_to_execution(set_status_to_execution)
    execution(execution)
    tools_approval_entry_executor(tools_approval_entry_executor)
    tools_approval_check_executor(tools_approval_check_executor)
    execution_tools(execution_tools)
    execution_supervisor(execution_supervisor)
    execution_handover(execution_handover)
    git_actions(git_actions)
    __end__([<p>__end__</p>]):::last
    __start__ --> build_context;
    build_context -. &nbsp;HandoverAgent&nbsp; .-> build_context_handover;
    build_context -. &nbsp;call_tool&nbsp; .-> build_context_tools;
    build_context -. &nbsp;stop&nbsp; .-> plan_terminator;
    build_context_handover --> planning;
    build_context_tools -.-> build_context;
    build_context_tools -. &nbsp;stop&nbsp; .-> plan_terminator;
    execution -. &nbsp;HandoverAgent&nbsp; .-> execution_handover;
    execution -. &nbsp;PlanSupervisorAgent&nbsp; .-> execution_supervisor;
    execution -. &nbsp;call_tool&nbsp; .-> execution_tools;
    execution -. &nbsp;stop&nbsp; .-> plan_terminator;
    execution -. &nbsp;tools_approval&nbsp; .-> tools_approval_entry_executor;
    execution_handover --> git_actions;
    execution_supervisor --> execution;
    execution_tools --> execution;
    planning -. &nbsp;stop&nbsp; .-> plan_terminator;
    planning -. &nbsp;PlanSupervisorAgent&nbsp; .-> planning_supervisor;
    planning -. &nbsp;HandoverAgent&nbsp; .-> set_status_to_execution;
    planning -. &nbsp;call_tool&nbsp; .-> update_plan;
    planning_supervisor --> planning;
    set_status_to_execution --> execution;
    tools_approval_check_executor -. &nbsp;back&nbsp; .-> execution;
    tools_approval_check_executor -. &nbsp;continue&nbsp; .-> execution_tools;
    tools_approval_check_executor -. &nbsp;stop&nbsp; .-> plan_terminator;
    tools_approval_entry_executor -. &nbsp;back&nbsp; .-> execution;
    tools_approval_entry_executor -. &nbsp;continue&nbsp; .-> tools_approval_check_executor;
    update_plan --> planning;
    git_actions --> __end__;
    plan_terminator --> __end__;
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
```
