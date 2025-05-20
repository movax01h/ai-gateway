require "gitlab-dangerfiles"

Gitlab::Dangerfiles.for_project(self, 'ai-gateway') do |dangerfiles|
  # Import all plugins from the gem
  dangerfiles.import_plugins

  # First-match win, so be sure to put more specific regex at the top...
  dangerfiles.config.files_to_category = {
    [%r{\Aduo_workflow_service/}, %r{(DuoWorkflowInternalEvent|InternalEventsClient|track_event)}] => [:duo_workflow_service, :ai_gateway, :analytics_instrumentation],
    %r{\Aduo_workflow_service/} => [:duo_workflow_service, :ai_gateway],
    %r{\Aclients/} => [:duo_workflow_service, :ai_gateway],
    %r{\Acontract/} => [:duo_workflow_service, :ai_gateway],
    %r{\Alib/} => [:duo_workflow_service, :ai_gateway],
    %r{\Aconfig/events/} => [:analytics_instrumentation],
    [%r{.*}, %r{(InternalEventsClient|track_event)}] => [:ai_gateway, :analytics_instrumentation],
    %r{.*} => :ai_gateway
  }.freeze

  # Import a defined set of danger rules
  dangerfiles.import_dangerfiles(only: %w[roulette type_label subtype_label z_retry_link])
end
