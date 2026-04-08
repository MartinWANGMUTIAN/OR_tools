---
name: awesome-design-md
description: Use curated DESIGN.md files from real websites to drive UI implementation, visual direction, and style extraction.
metadata:
  short-description: Apply website-inspired DESIGN.md files to frontend work
---

# Awesome DESIGN.md

Use this skill when the user wants a UI to match or borrow the visual system of a known website, or when you need a strong visual direction before implementing frontend work.

This skill packages the `VoltAgent/awesome-design-md` repository as local reference material. The source files live under `resources/design-md/<brand>/`.

## When to use

- The user wants a page or component styled like a known brand such as Vercel, Linear, Notion, Stripe, Supabase, Cursor, or Airbnb.
- You need a concrete design system before building a landing page, dashboard, docs page, or marketing surface.
- You want to extract colors, typography, spacing, and component treatments from an existing DESIGN.md instead of inventing a style from scratch.

## Workflow

1. Identify the closest matching brand or visual direction.
2. Read the corresponding `resources/design-md/<brand>/DESIGN.md`.
3. If needed, inspect `preview.html` or `preview-dark.html` in the same folder for a quick visual catalog.
4. Apply the design system deliberately.

Focus on:

- Color roles and surface hierarchy
- Typography families, scale, and tracking
- Button, card, input, and navigation styling
- Spacing rhythm, radius scale, and shadow system
- Responsive behavior and explicit design dos and don'ts

## Usage guidance

- Preserve the user's product structure and content. Borrow the visual language, not the brand identity.
- Reuse the design principles and tokens, but do not misrepresent the output as the original company or product.
- If the user names a brand that exists in `resources/design-md`, prefer reading that file directly.
- If the user asks for a vibe rather than a brand, choose the closest reference and say which one you used.
- For implementation tasks, translate the DESIGN.md into concrete CSS variables, component rules, and layout decisions.

## Finding references

Available references are stored at `resources/design-md/`. Search them with fast file tools, for example:

```bash
find resources/design-md -maxdepth 2 -name DESIGN.md
```

If multiple brands fit, compare 2 or 3 relevant DESIGN.md files and pick the strongest match instead of averaging them together.
