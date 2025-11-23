# Chain-of-Thought (CoT) Prompting Guide

> Source: https://www.promptingguide.ai/techniques/cot

## Overview

Chain-of-Thought (CoT) prompting enables complex reasoning capabilities through intermediate reasoning steps (Wei et al., 2022). This technique allows LLMs to break down complex problems into manageable reasoning steps before arriving at conclusions.

## Core Concept

CoT prompting works by demonstrating step-by-step reasoning. Rather than jumping directly to answers, models are prompted to articulate their logical process, significantly improving accuracy on reasoning-dependent tasks.

## Key Variants

### 1. Few-Shot CoT

Combines chain-of-thought with multiple examples showing the reasoning pattern.

**Example:**
```
Q: The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.
A: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False.

Q: The odd numbers in this group add up to an even number: 17, 10, 19, 4, 8, 12, 24.
A: Adding all the odd numbers (17, 19) gives 36. The answer is True.

Q: The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.
A: [Model completes with reasoning]
```

### 2. Zero-Shot CoT

A simpler approach using the magic phrase:

```
"Let's think step by step."
```

This surprisingly effective phrase alone can trigger reasoning without requiring multiple examples.

**Example:**
```
Q: I went to the market and bought 10 apples. I gave 2 apples to the neighbor
   and 2 to the repairman. I then went and bought 5 more apples and ate 1.
   How many apples did I remain with?

A: Let's think step by step.
   1. Started with 10 apples
   2. Gave 2 to neighbor: 10 - 2 = 8
   3. Gave 2 to repairman: 8 - 2 = 6
   4. Bought 5 more: 6 + 5 = 11
   5. Ate 1: 11 - 1 = 10

   Answer: 10 apples
```

### 3. Automatic Chain-of-Thought (Auto-CoT)

An automated approach with two stages:
1. **Clustering** - Group questions by diversity
2. **Sampling** - Generate reasoning chains automatically using zero-shot prompting

## Best Practices

1. **Demonstrate reasoning explicitly** in examples
2. **Keep demonstrations diverse** to cover varied problem types
3. **Use the universal trigger phrase** ("Let's think step by step") when examples aren't available
4. **Generate demonstrations automatically** when manual creation becomes impractical

## When to Use CoT

- Mathematical reasoning
- Multi-step logic problems
- Decision making with multiple factors
- Complex analysis tasks
- Any task requiring explicit reasoning trail

## Practical Impact

Standard prompting often fails complex problems, while CoT prompting successfully solves them through explicit step-by-step reasoning. Research shows improvements across benchmarks:
- GSM8K: +17.9%
- SVAMP: +11.0%
- AQuA: +12.2%
- StrategyQA: +6.4%

## Important Notes

- CoT works best with larger models (100B+ parameters)
- Smaller models may actually perform worse with CoT
- Zero-shot CoT may reduce performance on some task types
- Always test both approaches for your specific use case
