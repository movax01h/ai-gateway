"""Pattern tables for bl_discover_and_cluster; data only."""

import re
from typing import Dict, List

__all__ = ["ANCHOR_RES", "AUTHZ_GLOBS", "NOISE_SUFFIX", "ROLE_SEGMENTS"]

# Content rules (these tables and the tool's content net) must be justifiable
# from framework documentation, never from one repository's layout.

# ---------------------------------------------------------------------------
# AUTHZ_GLOBS -- the candidate net
# ---------------------------------------------------------------------------
# Recall-first: every pattern is fired unconditionally (no stack sniffing); the
# noise filter and clustering do the narrowing. A pattern that names a directory
# carries `**` before its basename half, so it covers the whole tree under that
# directory, not its top layer.
#
# Adding a pattern is not free: growing the pool is not monotone on the reviewed
# prefix. Argue a new glob at the shipped budget, per repository, with its
# candidate cost stated.
AUTHZ_GLOBS: List[str] = [
    # Rails
    "*_controller.rb",
    "app/controllers/**/*",
    "*policy*.rb",
    "app/policies/**/*",
    "*_resolver.rb",
    "app/graphql/**/*",
    "*finder*.rb",
    "app/finders/**/*",
    "*service*.rb",
    "app/services/**/*",
    "*serializer*.rb",
    "*ability*.rb",
    "*model*.rb",
    "app/models/**/*",
    # Django / Flask
    "*views.py",
    "*viewsets.py",
    "*/views/**/*.py",
    "*/api/**/*.py",
    "*/resources/**/*.py",
    "*permission*.py",
    "*serializers.py",
    "*models.py",
    "*/models/**/*.py",
    "*/graphql/**/*.py",
    # Express / Nest
    "*controller*.ts",
    "*controller*.js",
    "*route*.ts",
    "*route*.js",
    "*.guard.ts",
    "*middleware*",
    "*/routes/**/*",
    # Go; `*.go` is the real net and subsumes the directory globs.
    "*handler*.go",
    "*route*.go",
    "*/routers/**/*.go",
    "*/models/**/*.go",
    "*/services/**/*.go",
    "*.go",
    # Elixir, language-wide like `*.go`.
    "*.ex",
    # Rust, language-wide: Actix/Axum/Rocket/warp bind routes with macros or
    # builder calls, not a file-name convention, so no narrower glob exists.
    # No `ANCHOR_RES` entry; the semantic tier carries these files.
    "*.rs",
    # Java: Spring @Controller, JAX-RS *Resource, Struts *Action, Servlet API
    # *Servlet / *Filter; *.groovy covers Grails.
    "*Controller.java",
    "*Resource.java",
    "*Action.java",
    "*Servlet.java",
    "*Filter.java",
    "*Security*.java",
    "*Permission*.java",
    "*Authoriz*.java",
    "*Authentic*.java",
    "*AccessControl*.java",
    # `*.java` catches stacks that route by METHOD name, whose handlers carry
    # ordinary class names; no `ANCHOR_RES` entry for the same reason as Rust.
    "*.java",
    "*.groovy",
    # .NET: controllers by the "Controller" suffix; authorization as filters
    # and attributes.
    "*Controller.cs",
    "*Permission*.cs",
    "*Authoriz*.cs",
    "*Authentic*.cs",
    "*Security*.cs",
    "*Filter*.cs",
    "*Service.cs",
    "*AccessControl*.cs",
    # PHP: Laravel's app/Http/Controllers, app/Policies, app/Http/Middleware and
    # routes/*.php; Symfony's FooVoter.php.
    "*Controller.php",
    "*Voter.php",
    "*Policy*.php",
    "*Middleware*.php",
    "*Permission*.php",
    "*Authoriz*.php",
    "*Security*.php",
    "routes/**/*.php",
    # Django/Flask auth helpers (auth backends, Flask-Login blueprints).
    "*auth.py",
    # Language-agnostic role directories: the ROLE_SEGMENTS nouns with any
    # parent and any extension. Kept only where the convention is observable in
    # more than one project.
    "*/controllers/**/*",
    "*/services/**/*",
    "*/api/**/*",
    "*/endpoints/**/*",
    "*/handlers/**/*",
    "*/models/**/*",
    "*/policies/**/*",
    "*/middlewares/**/*",
    "*/rest/**/*",
    "*/schemas/**/*",
    "*/functions/**/*",
    # PSR-4 PHP extension points (naming convention, not a measurement).
    "*Manager.php",
    "*Plugin.php",
    "*Backend.php",
    "*Service.php",
]


# Noise file suffixes (bl_discovery's noise filter).
NOISE_SUFFIX = re.compile(
    r"("
    r"_spec\.rb|_test\.rb|\.test\.(js|ts|jsx|tsx)|\.spec\.(js|ts|jsx|tsx)|"
    r"_test\.go|_mock\.go|\.pb\.go|"
    r"\.md|\.min\.js|\.min\.css|\.bundle\.js|\.bundle\.css|\.map|"
    # Presentation layer: it renders decisions taken in server-side logic.
    r"\.tmpl|\.html|\.htm|\.vue|\.css|\.scss|\.less|"
    r"\.png|\.jpg|\.jpeg|\.gif|\.svg|\.ico|\.woff|\.woff2|\.ttf|\.eot|\.pdf"
    r")$"
)


# Request entry points (anchors), matched with `search` over the repo-relative path.
ANCHOR_RES = [
    re.compile(r"_controller\.rb$"),
    re.compile(r"(^|/)app/controllers/"),
    re.compile(r"_resolver\.rb$"),
    re.compile(r"(^|/)app/graphql/"),
    re.compile(r"views\.py$"),
    re.compile(r"viewsets\.py$"),
    re.compile(r"(^|/)views/.*\.py$"),
    re.compile(r"(^|/)api/.*\.py$"),
    re.compile(r"(^|/)resources/.*\.py$"),
    re.compile(r"(^|/)graphql/.*\.py$"),
    re.compile(r"controller.*\.(ts|js)$"),
    re.compile(r"route.*\.(ts|js)$"),
    re.compile(r"(^|/)routes/"),
    re.compile(r"handler.*\.go$"),
    re.compile(r"route.*\.go$"),
    re.compile(r"(^|/)routers/.*\.go$"),
    # Java (Spring MVC / JAX-RS / Struts / Servlet-API entrypoints)
    re.compile(r"Controller\.java$"),
    re.compile(r"Resource\.java$"),
    re.compile(r"Action\.java$"),
    re.compile(r"Servlet\.java$"),
    # .NET (ASP.NET MVC/Web API)
    re.compile(r"Controller\.cs$"),
    re.compile(r"(^|/)Controllers/.*\.cs$"),
    # PHP (Laravel/Symfony)
    re.compile(r"Controller\.php$"),
    re.compile(r"(^|/)Controllers/.*\.php$"),
    re.compile(r"(^|/)routes/.*\.php$"),
    # CanCan/CanCanCan: ability.rb declares every rule `authorize!` resolves
    # against, so it is an entry point in its own right.
    re.compile(r"(^|/)ability\.rb$"),
    # Elixir / Phoenix: controllers, the router.ex route table, and Plug modules
    # (plug/ or plugs/), where a Phoenix app installs authorization.
    re.compile(r"_controller\.ex$"),
    re.compile(r"(^|/)controllers/.*\.ex$"),
    re.compile(r"(^|/)router\.ex$"),
    re.compile(r"(^|/)plugs?/.*\.ex$"),
]


# Semantic roles by path segment; see bl_discovery's SEMANTIC surface.
ROLE_SEGMENTS: Dict[str, frozenset] = {
    # Data/ownership invariants -> IDOR/BOLA, mass-assignment allowlists.
    "state": frozenset(
        "model models entity entities domain schema schemas store stores storage repository "
        "repositories dao record records db database dal persistence orm".split()
    ),
    # Multi-step business rules and state transitions -> workflow bypass, TOCTOU.
    "logic": frozenset(
        "service services usecase usecases interactor interactors workflow workflows job jobs "
        "task tasks worker workers logic business core manager managers operation operations "
        "command commands transaction transactions process pipeline".split()
    ),
    # Explicit access-control machinery -> missing/improper authorization.
    "access": frozenset(
        "auth authz authn authentication authorization permission permissions policy policies "
        "acl rbac role roles guard guards middleware security access session sessions token "
        "tokens credential credentials identity principal".split()
    ),
}
