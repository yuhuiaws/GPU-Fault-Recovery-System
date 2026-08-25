# HTML build assets

`build-html.sh` downloads the Mermaid browser runtime when it is missing.
The generated JavaScript bundle is intentionally ignored by Git.

Pinned dependency:

- Package: Mermaid
- Version: `11.16.0`
- Source: `https://cdn.jsdelivr.net/npm/mermaid@11.16.0/dist/mermaid.min.js`
- SHA-256:
  `74d7c46dabca328c2294733910a8aa1ed0c37451776e8d5295da38a2b758fb9b`
- License: MIT, reproduced in `MERMAID-LICENSE.txt`

The generated `GPU_FAILURE_AUTOMATION_DESIGN.html` is a local build product.
Publish it through GitHub Releases, Pages, or CI artifacts rather than
committing it to the source tree.
