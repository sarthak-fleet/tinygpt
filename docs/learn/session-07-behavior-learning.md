# Session 7 — How models learn behavior (supervised, imitation, reinforcement)

> Sessions 1–6 covered models that predict the next TOKEN. But agents
> (NPCs in a game, robots, game-playing AIs) don't predict tokens — they
> pick ACTIONS. This session covers how models learn to act, not just to
> write. Direct application to Aliveville's NPC AI architecture.

## Where we are

A language model's job is: given the text so far, predict the next token.
An agent's job is: given the situation, pick an action. Both are
"functions on data" — the math we built in Sessions 1–5 still applies —
but the *training signal* is different.

This session covers the three main paradigms for learning behavior:

1. **Supervised / imitation** — train on (state, correct_action) pairs.
2. **Reinforcement** — learn from reward signals.
3. **Hybrid** — almost every modern agent combines both.

---

## The setup: agents and the decision loop

An agent lives in a loop:

```
state₀ → (pick action) → state₁ → (pick action) → state₂ → ...
                  ↑                          ↑
              decision                  decision
```

- **state** = what the world looks like right now (game position, recent
  dialog, sensor readings)
- **action** = what the agent does (move, speak, attack, emit a token)
- **transition** = how the action changes the state (game physics,
  conversation continuing, environment)

The agent's "model" — what we train — is a function from state to
action. The question is: how do we train it?

Language models live inside this loop too. The "state" is the
conversation so far; the "action" is which next token to emit. Same
loop, different state/action spaces.

---

## Paradigm 1: Supervised / Imitation learning

> **Collect (state, action) pairs from a demonstrator (often a human
> expert). Train a model to predict the action given the state.**

Mathematically identical to supervised learning. The "label" is just an
action instead of a class or a token.

Concrete instances:

- **Chess.** Collect 10,000 human chess games. Each move: `(board_state,
  move_played)`. Train a model to predict the played move given the
  board.
- **Game footage.** Record 100 hours of player gameplay. Each frame:
  `(game_state, player_input)`. Train.
- **Chatbot SFT.** Have humans write ideal chatbot responses. Each turn:
  `(conversation_so_far, ideal_response)`. Train. **This is SFT.** It's
  just behavioral cloning where the "behavior" is "respond helpfully."
- **Driving demos.** Record hours of expert driving. Each frame:
  `(sensor_readings, steering_inputs)`. Train.

What you get: a model that mimics the demonstrator. Quality of model =
quality of demos. If demos were good, model is competent. If demos were
mediocre, model is mediocre.

This is sometimes called **behavioral cloning** (BC) or **imitation
learning** when explicitly framed as "copy what a human did." All the
same thing mathematically.

---

## The compounding error problem (imitation's central flaw)

Pure behavioral cloning has a devastating failure mode. The setup:

- Training data is from a good demonstrator who rarely made mistakes.
- The model learns to act like the demonstrator in states the
  demonstrator saw.
- At deployment, the model makes small mistakes.
- Those mistakes put the model in states the demonstrator never reached.
- In those novel states, the model has no training signal — does
  something nonsensical.
- That nonsense produces even weirder states.
- Errors compound. The model drifts off-distribution and eventually does
  something catastrophic.

Classic example: a self-driving car trained purely on perfect human
driving demos doesn't know how to recover from being half-off-road —
because no demo ever drove half-off-road. The car drifts slightly, then
totally fails.

For language models the equivalent: a chatbot trained purely on perfect
human responses doesn't know how to recover from a conversation it
already messed up. Modern chatbots compensate with constant resets and
"sorry, let me start over" patterns.

Mitigations:
- **DAgger (Dataset Aggregation).** Iteratively collect demos in states
  the model actually reaches at deployment. Closes the gap between
  training distribution and deployment distribution.
- **Add noise to demos.** Train on slightly-degraded versions so the
  model sees off-distribution states during training.
- **Mix in RL.** Combine imitation with reinforcement learning so the
  model learns to recover from mistakes (next section).

---

## Paradigm 2: Reinforcement learning

> **Instead of `(state, correct_action)` pairs, you have an environment,
> a reward signal, and a policy. The agent acts, observes rewards,
> updates the policy to favor high-reward actions.**

The pieces:

- **Environment** — the world the agent acts in (a game, a simulation, a
  conversation with a human evaluator).
- **Policy** — the agent's model. `π(state) → action probabilities`.
- **Reward** — a number that says "this state is good" or "this action
  was good." Could be sparse (only at end of episode: +1 win, -1 lose)
  or dense (small signal at each step).
- **Trajectory** — a sequence of (state, action, reward, next_state)
  tuples generated by the policy interacting with the environment.

The training loop:

1. Policy acts in environment, generates trajectories.
2. Compute "return" for each step (cumulative discounted future reward).
3. Update policy: "make actions that led to high return more likely."
4. Repeat.

The math is more involved than supervised learning because you're not
just minimizing loss over fixed data. You're dealing with:
- **Exploration vs exploitation** — should I try new things or
  exploit known good actions?
- **Delayed rewards** — the action I took 100 steps ago might be what
  caused the reward I'm getting now.
- **Distribution shift** — as the policy changes, the data distribution
  changes (you're learning from data your current policy generates).

Two main families of RL algorithms:

### Q-learning

Learn a function `Q(state, action) =` "expected total future reward if
I take this action from this state." Pick actions that maximize Q.
Examples: DQN (deep Q-network, Atari benchmark), tabular Q-learning for
small problems.

Suited to discrete action spaces, well-defined state transitions.

### Policy gradient

Directly learn a policy `π(state) → action probabilities`. Update by
"make actions that led to high reward more probable, make actions that
led to low reward less probable." Examples: REINFORCE, A2C, **PPO**
(used in RLHF and OpenAI Five), TRPO.

Suited to continuous action spaces, where Q-learning becomes
infeasible. PPO is the modern workhorse — robust, easy to tune.

For our purposes, the family difference matters less than the core
insight: **RL learns from rewards, not labels.**

---

## When each paradigm fits

### Imitation works when:

| Condition | Why |
|-----------|-----|
| Lots of expert demos available | training signal is rich |
| Want human-like behavior (NPCs, dialogue) | mimicking IS the goal |
| Deployment closely matches training | minimal compounding-error risk |
| Need fast development cycle | imitation training is just supervised learning |

### Imitation fails when:

| Condition | Why |
|-----------|-----|
| Need superhuman performance | can't exceed demonstrations |
| Demos are scarce or noisy | not enough signal |
| Deployment will drift off-distribution | compounding errors |
| Need to discover novel strategies | model only knows what was shown |

### RL works when:

| Condition | Why |
|-----------|-----|
| Clean simulator + clean reward | RL is in its element |
| Need superhuman performance | RL can surpass demonstrations |
| Recovery from mistakes matters | RL learns it naturally |
| You have unlimited "training rollouts" cheap | RL is sample-hungry |

### RL fails when:

| Condition | Why |
|-----------|-----|
| Sparse rewards (signal only at episode end) | exploration is brutal |
| Hard to define reward | the whole approach depends on it |
| Real-world deployment | trial-and-error is costly |
| Limited simulator quality | RL exploits sim/real gap |

---

## The modern hybrid: warm-start + refine

Almost every successful modern agent combines both:

| System | Imitation stage | RL stage |
|--------|----------------|----------|
| **AlphaGo** | trained on human professional Go games | self-play RL refinement |
| **AlphaStar (StarCraft II)** | imitation on replay archives | self-play RL refinement |
| **OpenAI Five (Dota)** | minimal — mostly pure RL with curriculum + shaped rewards | (the main course) |
| **ChatGPT / Claude (RLHF)** | pretrain (SSL) + SFT (imitation on human demos) | PPO on human preference judgments |
| **Robotics arms (recent)** | sim + imitation from teleop | RL in sim, sim-to-real transfer |

The general pattern:
- **Imitation gives a competent baseline** — skipping random exploration
- **RL refines beyond that ceiling** — discovering strategies humans didn't

Neither alone gets you both fast convergence AND ability to surpass
demonstrations. The combination beats either component used alone.

---

## RLHF specifically (since this is what TinyGPT supports)

RLHF = Reinforcement Learning from Human Feedback. The three-stage
recipe used to turn a base LM into ChatGPT/Claude:

1. **Pretrain.** Self-supervised on web text. Model learns language
   structure. (TinyGPT's `tinygpt train` does this.)
2. **SFT (Supervised Fine-Tune).** Imitation on `(prompt, ideal_response)`
   pairs from humans. Teaches the assistant persona. (TinyGPT's `tinygpt
   sft` does this.)
3. **RL on preferences.** Humans rate pairs of responses; train a reward
   model on the ratings; then use PPO (policy gradient) to fine-tune
   the LM to produce responses the reward model rates highly. (TinyGPT's
   `tinygpt dpo` is a related but simpler variant.)

What gets learned at each stage:
- Stage 1: language and world facts.
- Stage 2: response shape ("answer the question, helpfully, in this
  style").
- Stage 3: subtle preferences ("not too short, not too long, refuse
  bad requests, prefer this phrasing").

The whole recipe is imitation → RL exactly as described above.

---

## For Aliveville specifically

Different NPC behavior layers want different approaches:

| Layer | Approach | Why |
|-------|----------|-----|
| **Dialogue / conversation** | LLM + prompt engineering | character description + memory in prompt; works at any quality |
| **Combat / decision-making** | Imitation from player footage | want HUMAN-LIKE NPCs, not optimal-but-weird |
| **World event responses** | LLM + state in prompt | flexible, plausible reactions to novel events |
| **Long-term character consistency** | Imitation + light RL | warm start from human play + reward shaping for "in character" behavior |
| **Quest dynamics** | Hand-coded + LLM | most rule-based, LLM for in-character interactions inside the rules |

**Two implications you might not have considered:**

1. **Imitation from your own players' footage = competitive moat.**
   Anyone can hit an OpenAI/Anthropic API; nobody else has YOUR players'
   actual playthroughs. NPCs trained on your data behave like Aliveville
   players, which is exactly the "alive" feel you want. This data
   becomes a primary asset of the product.
2. **The "perfect AI is uncanny" effect.** Players will find an NPC that
   plays "optimally" weird — too fast, too aggressive, too predictable
   in its weakness exploitation. NPCs that play "like a person who
   sometimes makes mistakes" feel real. The classic RL critique that
   "imitation can't surpass humans" becomes a feature here: you don't
   want to surpass humans, you want to mimic them well.

---

## Self-check

Don't peek:

1. **Why is pure behavioral cloning often a bad choice for self-driving
   cars but fine for chatbot SFT?**
2. **In RL terms, what's the "reward" in language-model RLHF?** (Hint:
   it's not "did the model output the right next token.")
3. **You want to train an NPC to behave like a wise old wizard. You have
   no game footage and no reward function — only 200 written
   conversations with this character.** What approach?
4. **Trap question:** if RL can surpass imitation, why do modern LLMs
   still rely on SFT (imitation) for the bulk of post-training, with
   only a thin RL layer on top?

---

## Where this connects

- Closes Session 4's brief mention of RL with substantive content.
- Behavioral cloning is *exactly* SFT — same machinery (Session 2's
  gradient descent), different data shape (state-action pairs instead
  of input-output pairs).
- Aliveville's NPC architecture is a direct application — imitation
  from player footage + prompt-engineered dialogue + light RL for
  refinement.
- Journal Entry 7 (paradigm taxonomy) is the high-level placement of
  these approaches; this session is the deep dive on the behavior-
  learning subset specifically.
