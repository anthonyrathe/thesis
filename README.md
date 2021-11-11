# Computer Science - Artificial Intelligence Master's Thesis: Benaderen van een strategie voor waardebeleggen met behulp van reinforcement learning en fundamentele data
English translation: Emulating a Value Investing Strategy with Reinforcement Learning and Fundamental Data
by Anthony Rathé

## About
This repository contains the source code used during the development of Anthony Rathé's Computer Science Master's thesis. The thesis text and presentation can be found in the root folder: RATHE-Anthony-thesis.pdf and presentatie.anthony.rathe.pdf respectively. Please note that both documents are in Dutch.

## Abstract (Dutch)
Het beheren van een beleggingsportefeuille biedt een uitdaging die institutionele en private beleggers al meer dan een eeuw in de ban houdt. Sinds enkele decennia worden ook datawetenschappen en artificiële intelligentie ingezet voor het oplossen van dit optimalisatieprobleem. Hierbij zijn echter zowel Reinforcement Learning
(RL) als het gebruik van fundamentele data in dit onderzoeksdomein eerder op de
achtergrond gebleven.

In deze thesis trachtten we dan ook met behulp van Reinforcement Learning en
fundamentele data een beleggingsstrategie te benaderen —meer specifiek die van het
waardebeleggen—, waarbij een agent een portefeuille bestaande uit twaalf aandelen
en cash dient te optimaliseren d.m.v. long-only posities.

Hiertoe selecteerden we een set van twaalf fundamentele features en ontwikkelden
we vier RL agents waarin we al dan niet waardebeleggen-specifieke domeinkennis
integreerden. We testten deze agents voor 18 parameterconfiguraties en stelden
daarbij vast dat de voorgestelde agents —mits beperkte transactiekosten— succesvol
waren in het verslaan van de buy-and-hold benchmark (met statistische significantie).
De integratie van domeinkennis leverde hiertoe echter geen significante bijdrage,
wat —in combinatie met een over-sensitiviteit aan transactiekosten door hoger dan
verwachte transactievolumes— leidde tot de conclusie dat we er niet in zijn geslaagd
specifiek de strategie van het waardebeleggen te benaderen.

We hebben met dit onderzoek echter wel aangetoond dat fundamentele data
en Reinforcement Learning met succes kunnen worden toegepast op het portfoliooptimalisatieprobleem
en hopen met toekomstig werk de hoge transactievolumes te
kunnen mitigeren, wat moet leiden tot meer robuuste performantie onder toenemende
transactiekosten.

## Major folders, files and classes
In order to get familiar with the codebase for this thesis, it is recommended to have a look at the following folders, files and classes in the presented order:
1. **src/agents/experiment.py**: this file contains the framework for running an experiment as described in the thesis text. In this script we construct the agent, which consists of: a Reinforcement Learning algorithm (for which we use the **stabele baselines** package), an environment in which the agent generates actions and is rewarded and a policy (in essence a neural network) that will determine how the agents transforms the state of the environment into an optimal next action.
2. **src/agents/experiment_1_a_new.py** and similar: these files contain the actual instances of the experiments as described in the thesis text.
3. **src/environments**: this folder contains the 3 environments used in this thesis, in which the agent is asked to
   - provide a continuous portfolio vector (**StackedEnv.py**)
   - provide a binary portfolio vector (**StackedEnvBinary.py**)
   - provide a continuous portfolio "delta" (**StackedEnvDiff.py**)
   - (a fourth environment developed for experimentation only)
Each environment makes use of a simulator (**./simulators/BasicSimulator.py**) that simulates a portfolio based on portfolio vectors provided at regular intervals and actual historical data.
4. **src/policies/SharedStackedPolicy.py**: this file contains the custom parametrised policy made for this thesis, which allows the user to specify a neural network architecture that takes subsets of the input into identical subnetworks, which are merged into a shared network that eventually splits to produce the expected reward and the expected optimal action (please refer to literature on Proximal Police Optimization for more insights on policy networks)

