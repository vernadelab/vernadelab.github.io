---
layout: page
title: research
permalink: /research/
description: The foundations of learning and decision-making over time.
nav: true
nav_order: 2
---

We study agents that act, learn from feedback, and carry useful experience from one task to the next. Our aim is to connect rigorous theory with the demands of genuinely adaptive systems.

## Current research programmes

{% assign sorted_projects = site.projects | sort: "importance" %}

<div class="mb-4">
  {% for project in sorted_projects %}
    <a class="btn btn-outline-primary" href="#{{ project.title | slugify }}">{{ project.title }}</a>
  {% endfor %}
</div>

{% for project in sorted_projects %}

<h3 id="{{ project.title | slugify }}">{{ project.title }}</h3>

{{ project.content | markdownify }}

[Back to research programmes](#current-research-programmes)
{% endfor %}

## Research themes

### Memory and POMDPs

How can an agent make decisions when its observations reveal only part of the state? We study when it can act on the current observation alone and what it should remember when it needs more context.

**People:** Onno Eberhard and Amer Essakine.

- Eberhard, Vernade and Muehlebach. [Commit to the Bit: Reactive Reinforcement Learning Done Right](https://arxiv.org/abs/2605.28276) (ICML 2026).
- Eberhard, Muehlebach and Vernade. [Partially Observable Reinforcement Learning with Memory Traces](https://arxiv.org/abs/2503.15200) (ICML 2025).

### Non-stationarity in bandits, reinforcement learning, and robotics

We study how bandit algorithms detect shifts in rewards and how reinforcement learning and control can help robots adapt during online interaction.

**People:** Nicolas Nguyen (bandits) and Marko Zarić (reinforcement learning and robotics).

- Nguyen, Gaucher and Vernade. [Non-Stationary Lipschitz Bandits](https://arxiv.org/abs/2505.18871) (NeurIPS 2025).
- Eberhard, Vernade and Muehlebach. [A Pontryagin Perspective on Reinforcement Learning](https://arxiv.org/abs/2405.18100) (L4DC 2025).

### Imitation learning and multi-objective imitation

We study what demonstrations reveal about how a learner makes decisions and how to learn from several experts whose objectives differ.

**People:** Michela Petriconi and Ziyad Sheebaelhamd (imitation learning), with Ziyad focusing on multi-objective imitation.

- Petriconi, Menet and Vernade. [Learning from a Bandit Learner](https://arxiv.org/abs/2610.26103) (2026 preprint).
- Sheebaelhamd, Viano, Cevher and Vernade. [Split the Differences, Pool the Rest: Provably Efficient Multi-Objective Imitation](https://arxiv.org/abs/2605.12000) (NeurIPS 2026).

### Risk-aware optimization in MDPs and safe reinforcement learning

We study how to plan and learn in Markov decision processes (MDPs) when poor outcomes matter, and how agents can satisfy safety constraints even under disturbances.

**People:** Alexandre Marthe and Amer Essakine (risk-aware optimization), and Alexander von Rohr and Gloria Desideri (safe reinforcement learning).

- Marthe, Bounan, Garivier and Vernade. [Efficient Risk-sensitive Planning via Entropic Risk Measures](https://arxiv.org/abs/2502.20423) (2025).
- Massiani, von Rohr, Haverbeck and Trimpe. [Viability of Future Actions: Robust Safety in Reinforcement Learning via Entropy Regularization](https://arxiv.org/abs/2506.10871) (ECML PKDD 2025).

### Testing and evaluation

We study how to draw reliable conclusions from limited or adaptively collected data, including when there is enough evidence to stop a sequential test.

**People:** Amin Charusaie and Adrienne Tuynman.

- Tuynman and Mathieu. [Optimal Posterior E-values with Non-Convex Parameter Sets with Applications to Voting Systems](https://arxiv.org/abs/2606.29998) (2026 preprint).

### Theory of continual learning and continual reinforcement learning

We ask what an agent should retain from earlier tasks, when it should adapt to new conditions, and how to measure success over a stream of tasks.

**People:** Gloria Desideri and Thomas Grote.

- Lanzillotta, Samiei, Precup, Pascanu and Vernade. [To Retain or to Adapt? Generalizing Continual Learning](https://arxiv.org/abs/2607.05609) (TMLR 2026).
