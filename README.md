# spec_driven_xpbd

**A method for steering coding agents on underspecified tasks, validated by a
physics engine in a domain I knew nothing about.**

As an ML researcher with zero prior knowledge of physics engines, I built a
working [XPBD rigid body
simulator](https://matthias-research.github.io/pages/publications/PBDBodies.pdf)
in JAX in under 20 hours using Claude Sonnet 4.6. The physics engine is the
proof of concept. The actual contribution is the workflow.


## The Problem: Coding Agents Silently Fill Gaps

When tasks are underspecified, and they always are in a new domain, coding
agents make silent assumptions. Those assumptions compound. By the time you
notice, you've built the wrong thing.

## The Technique: Maieutic Prompting

*Maieutics* is the Socratic method of drawing out latent ideas through
questioning. Applied to coding agent-assisted development:

> Interview the user relentlessly until the task is fully determined. Never
> implement under ambiguity.

This is encoded as a 5th rule in [`CLAUDE.md`](./CLAUDE.md), extending
[andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills)'s
four-rule structure. The interview pattern itself is inspired by
[grill-with-docs](https://github.com/mattpocock/skills/blob/main/skills/engineering/grill-with-docs/SKILL.md)
by Matt Pocock.

So it becomes the agent's responsibility to:
 
1. interview you until specifications is unambiguous (and if you don't
   understand the question you can always ask !),
2. manage and persist those specifications,
3. implement.


### A concrete example

Standard XPBD implementations use a Gauss-Seidel (GS) solver. I didn't know
this. When I described my performance requirements, Claude asked:

> *"Do you need the solver to converge to high precision, or is approximate but
> fast convergence acceptable?"*

That question surfaced a constraint I hadn't articulated. The answer led Claude
to propose a Jacobi solver which became the default. GS would have been slower
and unnecessarily precise for real-time simulation. I would never have caught
this on my own.

## The Artifact: SPECIFICATIONS.md

As the project evolves, every clarified decision gets written into
[`SPECIFICATIONS.md`](./SPECIFICATIONS.md) — a living document now over 500
lines long. It records every specific choices that was made.

**The test:** an independent agent with only `SPECIFICATIONS.md` should be able
to reconstruct the project to its current state.


## Results

| Scene | What it tests |
|-------|--------------|
| `box-arena` | Collision detection and resolution between 10 dynamic bodies |
| `double-pendulum` | Joint constraints, chaotic dynamics |


![box-arena](assets/box_arena.gif)

![double-pendulum](assets/double_pendulum.gif)


## Workflow

```
A. Init
  1. Research domain, save docs
  2. Interview to initialize SPECIFICATIONS.md
  3. High-level plan with checkboxes in SPECIFICATIONS.md

B. Per task in the high-level plan
  1. Fresh context, load SPECIFICATIONS.md + relevant docs
  2. Interview → precise spec for this task
  4. Update SPECIFICATIONS.md
  3. Implement, then test, then human verification
  4. Update SPECIFICATIONS.md
  5. Commit
```

Clear context between tasks forces the spec to be the single source of truth
rather than accumulated conversation history.


## Getting Started

```shell
uv run xpbd --help
uv run xpbd box-arena
uv run xpbd double-pendulum
```

The project defaults to CPU. For GPU support, edit `pyproject.toml` with your [JAX version](https://docs.jax.dev/en/latest/installation.html).


## Why This Matters for AI Engineering

The hard part of building with coding agents is **decomposing ambiguous goals
into verifiable specs**. This project is a concrete instance of doing that on a
non-trivial technical problem in an unfamiliar domain.
