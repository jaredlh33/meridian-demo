# Security Policy

## Reporting a vulnerability

**Use GitHub's private vulnerability reporting only:** open the
[Security tab](https://github.com/jaredlh33/meridian-demo/security/advisories/new)
of the `jaredlh33/meridian-demo` repository and choose **"Report a
vulnerability"**. Reports stay private until a fix is ready — please do
not open a public issue, pull request, or discussion about a
vulnerability.

There is no email address or other reporting channel for this project.
If private vulnerability reporting is ever disabled on the repository,
the policy here will be updated to say so.

## What happens next

- The maintainer triages the report and confirms whether it reproduces.
- If it is a real vulnerability, a fix is prepared privately, then
  released with a note crediting the reporter (unless the reporter
  prefers to stay anonymous).
- There is no bounty program and no promised response time; this is a
  solo-maintainer project, but security reports are treated as the
  highest-priority work.

## Scope notes

- The demo backend at
  [`meridian-demo-3z7h.onrender.com`](https://meridian-demo-3z7h.onrender.com)
  runs **no secrets and no API keys**: it serves local GGUF models on
  CPU, keeps no user data, and never logs prompts (only a request
  counter is kept). There is nothing to leak from it by design.
- The router generates a fresh ephemeral Ed25519 session key on every
  start; there is no long-lived credential anywhere in the project.
- The website is a static site; it holds no user data.
