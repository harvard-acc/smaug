"""Downloads build artifacts via the CircleCI API."""

import argparse
import os
import json
import six
import subprocess
import sys

def query_artifacts(api_token, user, project, branch, filter):
  """Queries CircleCI for a list of build artifacts.

  All parameters are strings. The returned value is a JSON object with the
  following attributes:
    "path": Path to the artifact in the project, relative to the working
            directory.
    "pretty_path": Same as path.
    "node_index": Ignored.
    "url": The URL at which the artifact is located.
  """
  url = "https://circleci.com/api/v1.1/project/github/%(user)s/%(project)s/latest/artifacts?branch=%(branch)s&filter=%(filter)s" % {
      "user": user,
      "project": project,
      "branch": branch,
      "filter": filter,
      }
  args = ["curl",  "-H'Circle-Token: %s'" % api_token, url]
  print(" ".join(args))
  proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = proc.communicate()
  print("Stdout: %s", stdout)
  print("Stderr: %s", stderr)
  if proc.returncode != 0:
    raise OSError("artifact lookup failed")
  artifacts = json.loads(stdout)
  return artifacts


def download_artifacts(artifacts, artifact_name, download_loc):
  """Downloads matching artifacts to the desired location.

  artifacts: List of artifacts from the CircleCI API.
  artifact_name: The basename of the artifacts to download.
  download_loc: The directory where the objects will be downloaded to.
  """
  for artifact in artifacts:
    if os.path.basename(artifact["path"]) == artifact_name:
      proc = subprocess.Popen(["wget", "-P", download_loc, artifact["url"]])
      stdout, stderr = proc.communicate()
      if stdout is not None:
        print(six.ensure_str(stdout))
      if stderr is not None:
        print(six.ensure_str(stderr))
      if proc.returncode != 0:
        raise OSError("failed to download artifact")


def main():
  parser = argparse.ArgumentParser(
      "Download CircleCI artifacts for SMAUG dependencies")
  parser.add_argument("--artifact_name", type=str, required=True,
                      help="Base filename of the object to download.")
  parser.add_argument("--download_loc", type=str, required=True,
                      help="Directory to store the downloaded artifact.")
  parser.add_argument("--api_token", type=str, required=True,
                      help="Secret API token for access to CircleCI APIs.")
  parser.add_argument("--project", type=str, required=True,
                      help="Name of the project to download from")
  parser.add_argument("--user", type=str, default="harvard-acc",
                      help="CircleCI username.")
  parser.add_argument("--branch", type=str, default="master",
                      help="Branch from which the artifact was built.")
  parser.add_argument("--filter", type=str, default="successful",
                      help="Filter on which builds to return.")
  args = parser.parse_args()
  artifacts = query_artifacts(args.api_token,
                              args.user,
                              args.project,
                              args.branch,
                              args.filter)
  if "message" in artifacts:
    print("Artifact query failed:", artifacts["message"])
    sys.exit(1)
  download_artifacts(artifacts, args.artifact_name, args.download_loc)


if __name__ == "__main__":
  main()
