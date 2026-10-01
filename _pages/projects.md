---
layout: page
title: research
permalink: /research/
description: The foundations of learning and decision-making over time.
nav: true
nav_order: 2
horizontal: true
---

We study agents that act, learn from feedback, and carry useful experience from one task to the next. Our aim is to connect rigorous theory with the demands of genuinely adaptive systems.

## Current research programmes

<div class="projects">
{% assign sorted_projects = site.projects | sort: "importance" %}
<div class="container">
  <div class="row row-cols-1 row-cols-md-2">
  {% for project in sorted_projects %}
    {% include projects_horizontal.liquid %}
  {% endfor %}
  </div>
</div>
</div>

## Research themes

- **Non-stationarity:** detecting and responding to a world that changes over time.
- **Exploration:** gathering the right information under partial, delayed, or structured feedback.
- **Risk and control:** moving beyond expected return toward reliable decision-making.
