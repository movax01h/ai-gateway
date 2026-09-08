require "gitlab-dangerfiles"

Gitlab::Dangerfiles.for_project(self, 'ai-gateway') do |dangerfiles|
  # Import all plugins from the gem
  dangerfiles.import_plugins

  # First-match win, so be sure to put more specific regex at the top...
  dangerfiles.config.files_to_category = {
    [%r{\Aduo_workflow_service/}, %r{(DuoWorkflowInternalEvent|InternalEventsClient|BillingEventsClient|track_event|track_billing_event)}] => [:duo_workflow_service, :ai_gateway, :analytics_instrumentation],
    %r{\Aduo_workflow_service/} => [:duo_workflow_service, :ai_gateway],
    %r{\Aclients/} => [:duo_workflow_service, :ai_gateway],
    %r{\Acontract/} => [:duo_workflow_service, :ai_gateway],
    %r{\Alib/} => [:duo_workflow_service, :ai_gateway],
    %r{\Aconfig/events/} => [:analytics_instrumentation],
    [%r{.*}, %r{(InternalEventsClient|track_event|track_billing_event|BillingEventsClient)}] => [:ai_gateway, :analytics_instrumentation],
    %r{\Aai_gateway/model_selection/models\.yml\z} => [:ai_gateway, :utilization],
    %r{.*} => :ai_gateway
  }.freeze

  # No teammate carries a `utilization` role in the roulette data, so that row
  # can only ever render as blank. Fulfillment approval on `models.yml` comes
  # from `.gitlab/CODEOWNERS`, not from this category, so dropping the row
  # costs nothing.
  dangerfiles.config.disabled_roulette_categories = [:utilization]

  # Import a defined set of danger rules
  dangerfiles.import_dangerfiles(only: %w[roulette type_label subtype_label z_retry_link large_diff])
end
