source 'https://rubygems.org'

gem 'gitlab-dangerfiles', '~> 4.10.0'
# The danger-review CI component resolves gems fresh each run (no committed
# Gemfile.lock). git 6.0.0 removed Git::Base and the Git::Log Enumerable
# interface, which danger 9.6.1 still relies on, crashing the job with
# `uninitialized constant Git::Base` (gitlab-org/ruby/gems/gitlab-dangerfiles#108).
# Cap `git` below 6.0.0 until danger ships a git-6-compatible release.
gem 'git', '< 6.0.0'
