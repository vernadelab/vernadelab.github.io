# Agent guidelines for the Vernade Lab site

This is the authoritative entry point for coding agents working in this repository. The guidance is adapted from the official al-folio agent instructions while respecting the version currently installed here.

## Editing priorities

1. Keep the site close to standard al-folio.
2. Prefer content and configuration changes in `_config.yml`, `_data`, `_pages`, `_projects`, `_news`, and `_bibliography`.
3. Use al-folio's existing layouts, includes, collections, and BibTeX support before adding HTML, Liquid logic, JavaScript, or Sass.
4. Do not create site-specific layouts, includes, or styles unless the requested result cannot be expressed with an existing template feature.
5. Preserve the existing content and URLs unless the user explicitly asks to remove or rename them.

## Where content lives

- Home page: `_pages/about.md`
- Team order and portraits: `_pages/profiles.md`
- Team biographies and links: `_pages/team/*.md`
- Research overview: `_pages/projects.md`
- Research programmes: `_projects/*.md`
- Publication page: `_pages/publications.md`
- Publication records: `_bibliography/rl-theory.bib` and `_bibliography/interactive-learning.bib`
- Teaching: `_pages/teaching.md`
- Applications FAQ: `_pages/join.md`
- News: `_news/*.md`
- Images: `assets/img/`

## Version boundary

The official al-folio v1 starter is plugin-based, but this site currently uses an earlier monolithic al-folio version. Do not copy v1 gem wiring or remove the installed `_layouts`, `_includes`, `_sass`, or JavaScript runtime as part of an ordinary content edit. A v1 upgrade must be handled as a separate migration with its own build and deployment validation.

## Validation

Run the following from the repository root after content changes:

```bash
bundle exec jekyll build
git diff --check
```

For dependency, template, or styling changes, also run the repository's formatting checks and inspect the home, people, research, publications, teaching, and join pages locally.
