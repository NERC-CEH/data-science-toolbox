import json
from sphinx.application import Sphinx

def inject_aws_button(app, pagename, _templatename, context, doctree):
  if not pagename.startswith("methods/"):
    return

  options = app.config.aws_launch_options
  if not options:
    return

  high_compute = False
  if doctree is not None:
    metadata = doctree.document.settings.env.metadata.get(pagename, {})
    raw = metadata.get("high_compute", False)
    high_compute = str(raw).lower() == "true"

  static_prefix = "../" * pagename.count("/")

  inline_vars = f"""
    <script>
      window.AWS_DOC_PATH = {json.dumps(pagename)};
      window.AWS_LAUNCH_OPTIONS = {json.dumps(options)};
      window.AWS_HIGH_COMPUTE = {str(high_compute).lower()};
      window.AWS_STATIC_PREFIX = {json.dumps(static_prefix)};
    </script>
    """
  context["body"] = context.get("body", "") + inline_vars

def setup(app: Sphinx):
  app.add_config_value("aws_launch_options", default=[], rebuild="html")
  app.connect("html-page-context", inject_aws_button)
  return {"version": "0.1", "parallel_read_safe": True}
