# POET/ePOET Algorithmic/Implementation Dive

This is a deep dive into the ideology behind POET/ePOET algorithms, with a focus on understanding followed by suggested improvements for learning/compute. The later sections are on implementation suggestions, which *do not* change the results of any algorithms. 


* **Initialization**: 
  * **Single start** - By default, POET starts with a single, random agent on a single, flat environment. The reasoning for this is that "it is intuitive". I don't see why we need a flat environment, instead of almost flat, or why we start with only 1 agent, instead of a full population. What if:  
    * We start with a full population of flat environments and random agents - This gives us many different starting points, one of which may be closer to walking than another, right from the start. Akin to using LHS to cover initial parameter space and speed up convergence.
    * We start with a full population of random agents and almost-flat environments - This provides some difference in environments, while remaining very simple. It could provide a more-robust starting state, or we could get transfers earlier and then agents that learn to walk faster. Or this is a terrible idea, no one knows.  
* **Reproduction**:
  * **POET Process**: 8 steps
    1. Check for environments that are "good enough" to reproduce (currently scoring over 200)
    2. Randomly generate 512 possible offspring from the list from **1**. 
    3. Check for MCC consistency (parent agent must score 50 < score  < 300)
    4. Rank by "Novelty". 
        * Novelty is defined as the average of the normalized Euclidean distance between the current environment and the 5 nearest neighbors (KNN where K=5), where the nearest neighbors are all existing environments, active and archived. 
    5. Perform transfer test using currently active agents, and keep best one.
        * This involves testing the agents as-is on the new environment, and then testing all of the agents again after 1 iteration of optimization. If there are `n` agents, that means `n` optimizations plus `2n` evaluations.
    6. Check for MCC consistency using currently-paired agent (i.e., the new "best" agent may find it too easy)
    7. Add new environments to the current pool, until some chosen limit is reached.
    8. Archive oldest environments (does not care if we beat them or not) until we're at max pool size.
  * **ePOET Process**: Steps 1-3 and 6-8 are the same.
    1. Same.
    2. Same.
    3. Same.
    4. Rank by "Novelty".
        * Novelty is now defined as "Performance of All Transferred Agents Environment Characterization" (PATA-EC).
          1. Score all agents, archived and active, against the new environment.
          2. Clip scores between 50 < score < 300. Argument is that too bad or too good is uninformative. Do we agree with this? Should this be built into rejecting environments prior to transfer test?
          3. Rank-normalize - replace scores with ranks, normalize between [-0.5, 0.5].
          4. Apply Euclidean distance on step **3**, using the same KNN approach as POET. They claim that this worked better than rank-based distance metrics.
    5. Peform transfer test.
        * Possible transfers are tested against the best of the 5 most recent scores from the existing agent.
        * New transfers are first tested as-is on the environment, but must pass the metric without optimization.
        * Any agents that passed the metric undergo 1 round of optimization, and are tested again. Again, they must pass the metric.
        * The best passing agent is accepted as the new transfer.
    6. Same
    7. Same.
    8. Same
  * **Improvements**: 
    * **Rate** - We only check every ~100 iterations (`optim_iters` * `reproduction_interval`), which lines up with transfers as well (to reduce duplicated work). Why not reproduce as soon as an agent/environment satisifies the criteria? Is there something special about the wait-time between reproductions and its effect on learning?
    Side remark, lining this up with transfers only reduces transfer compute on the new environment, because they perform a transfer step when first created. 
    * **Generation** - Currently, it generates 512 possible offspring and then tests all of them for several metrics. 
        1. We could put this into a while loop, and test one at a time, and accept environments if they pass all the criteria, and break when we have enough new offspring. 
        The ideological impact is on step **4**, where we would no longer rank by novelty, since there's nothing to rank against.
        2. Same while loop, but we could still measure novelty, and simply put a lower-bound on how novel we think new environments should be. This would involve still running the novelty ranking from step **4** above, just with some distance cuttoff.
        3. Same while loop, but define novelty relative to difficulty (more on this below). Basically, novelty is supposed to push agents to learn new skills and "grow". Creating more difficult environments is akin to testing skills/knowledge the agents don't have, *or have forgotton*. Skills agents don't have qualify as novelty, skills agents have forgotten need "relearned", so going back to that part of parameter space may not be a bad thing. This would rely on testing active agents only, no archived agents.
        Since novelty doesn't directly play into ANNECS, we wouldn't need to keep the old check at all here.
    * **Replacement Method 1** - When new environments are created, they replace the "oldest" environment in the group (probably a deque that's kept at a constant length). 
        1. Environments should be most related to the parents that they came from. Why not replace the parents with their children? 
        2. We calculate "novelty" when creating environments, why not use that to find the most related active environment? (I still expect this to be parents, but it could be another environment.)
        
        Will these ideas require a counter on "unsolved" environments to make sure they don't stick around forever?
    * **Replacement Method 2** - There isn't a maximum number of offspring permitted to any agent/environment pair. So, an agent that quickly solves its environment will reproduce at every possible time. Is this good for learning? Could limiting this be helpful? Should we scale by number of offspring already provided - this could maintain diversity, but idk what other effects it could have.
    * **Difficulty** - This uses MCC to define min/max difficulty of new environments based on current agent performance against them, then tested again after transfer has been performed to make sure the new "best" agent is still within bounds. A couple ideas:
        1. What if we make new environments less easy, so that each environment generates more learning? e.g., we call 230 solved and use 200 for reproduction, but, the MCC limit is 300, implying that some new environments may get created already solved (hopefully taken care of with the novelty ranking system currently in place). What if we define a heuristic, like, "an environment can be no easier than 90% of the reproduction threshold", some rule-of-them that works across a range of agents/environments?
        This would remove the necessity for a second MCC check after transfer.
        2. What if we make environments less hard? Increasing the lower end would generate smaller "steps" in learning, but they might be easier to optimize, so we'd take more, smaller steps compared to **1**.
        3. I think we need to make "solved" the same level as "reproduce". I can't find the reasoning for why we reproduce at 200, but are solved at 230, while the original BW considers 300 solved. 
    * **Novelty** - Novelty is handled by PATA-EC (in ePOET). What is the impact of relating "novelty" to "difficulty"? Will we lose our specialist agents and create more generalists? Can we train as fast or as far? Is a slower training rate fine with the reduced compute on reproduction? 
* **Transfers**:
  * **POET Process**:
    1. Score all agents as-is against an environment.
    2. Evolve all agents for 1 iteration on the new environment, and score them all again.
    3. Take best scoring agent.
        * If the new agent came from step **1**, it's a "direct" transfer
        * If the new agent came from step **2**, it's a "proposal" transfer
  * **ePOET Process**: 
    1. Score all agents as-is against an environment. Agents must beat the current agent's best score in the 5 most recent iterations
    2. Any agents passing the score are evolved for 1 iteration on the new environment and then scored again. This score must also pass the current agent's score.
    3. If any agents pass both rounds, take best scoring agent.
        * There is no distinction between "direct" and "proposal" transfer anymore, because both aspects are used.
  * **Improvements**: 
    * **Rate** - We only check every `optim_iters`, and then check each agent against all other active environments and compare against all active agents. Why not use a continuos, stochastic check? e.g., at every iteration, we test transfers against 1 or a few other environments, and if it passes, move them. This would require changing their competitive aspect to transfers, but is that a bad thing? It would prevent a single agent from sweeping through the entire population as quickly. 
    * **Function** - ~~The current transfer first tests all agents against an environment, then runs one ES step on agents that do well enough, and test those agents again, then keeps the best performing transfer agent (if any). The argument is that this conforms to the "exptected transfer rate" from the MCC paper. Why does this matter? How does this help? I like the single ES step performed, as it gives "one-shot" to improve on the environment, but that's also the most expensive part of the transfer. What if we drop the ES step, and just test if a new agent performs better than the current one? This is all just learning, so who cares if it is better now or after 1 learning step?~~
    I actually like how ePOET handles it, and don't really see a reason to change. 
* **Progress**:
  * **POET Process**: None
  * **ePOET Process**: ePOET uses "accumulated number of novel environments created and solved" (ANNECS). This metric involves two steps:
    1. Must pass all creation tests. So, it must be the product of a successful reproduction.
    2. It must eventually be solved (defined as a score >230 currently). This is not what's implemented in the code though - seems to just pass an MCC check, against an archived agent. I don't fully follow the final check.
  * **Improvements**:
    * ANNECS isn't a terrible idea, but there are 4 distinct endings to an environment.
      1. Your environment reproduces and it creates an agent worth transferring to another environment. This indicates mastery of the current environment and learned skills worth transferring.
      2. Your environment does not reproduce, but it does create an agent worth transferring. This indicates skills worth transferring.
      3. Your environment does reproduce, indicating mastery, but does not transfer an agent.
      4. Your environment never reproduces or transfers an agent. 

      These scenarios cover the 2 parts of "phylogenies" - genetic contributions to the population and cultural (e.g. learning) contributions to the population. The paper effectively suggests endings 1 and 3, where the environment is "solved", without paying attention to transfer events. However, these aren't checked in the code, so really all 4 options are retained, which is excessive and misleading. I suggest we modify thi to the following:
      1. We should define "solved" as "reproduce". The assumption in the paper is that an environment capable of reproducing will be soon solved. I think it's a fudge factor to make sure reproduction happens before environments are pushed out or over optimized. However, this does not imply cultural/learning contributions, as the new environment could get paired with a different agent before being added to the population.
      2. An environment that creates an agent worth transferring has provided cultural knowledge to the population, even if it does not reproduce and provide genetic contributions. 
      3. An environment that does not reproduce nor provide an out-bound transfer has contributed nothing to the population. This could be from difficulty (too hard to solve with the current skillset) or because it can't be solved (a cliff). This agent/environment should be thrown-out without record. If it's just too hard at the moment, it could be generated again later, when agents have matured. If it's impossible to solve, there's no reason to be testing on it, and it will get thrown out again next time it appears. 

      "Strong" solutions both reproduce and provide outbound transfer. "Weak" solutions either reproduce or provide outbound transfer. These correspond to combinations of **1** and **2** above. I'm not sure the best way to handle this. I think both are important, but it might also be prescient to acknowledge reproduction without inbound transfer, as this indicates novel skills that other agents can't replicate.        
      "No" solutions provide nothing, and correspond directly to point **3** above. 
* **Engineering**:
  * **Profiling** - Where are we actually spending time? The paper focuses on process communication overhead. Could it be model evaluation? Gym evaluation? Running a giant model (512x256x64 with relu activation between layers) is super slow, which leads me to believe that model eval is pretty costly at the moment (this is anectdotal at best). 
    * Could suggest `cython-ing` models or environments, for speed without leaving python
    * Many models use tensorflow or pytorch, so maybe worth implementing models in that? I'm not sure how well GPUs would help, as we don't have enough to really scale the way POET is designed, but maybe CPU implementations are also more efficient?
    * Can environments be simplified? E.g., what is necessary for training vs what looks pretty in videos, and does that matter at all? This won't scale to all environments, so agents are probably more important (again, can't know without profiling).
    * Is there a possibility for shared-memory parallelism? It would limit the memory/compute availability, but would abolish communication overhead. 
      * <span style="color:blue">2022/09/20 - Not really. Python GIL means we either get shared-memory with GIL or socket connections with low-copy but no GIL issues. `fiber` handles the socket connections with minimal memory copy, with the benefit of no GIL. Anything more "shared-memory" like would have to be done below python.</span>
  * **Fiber** - I know very little about how it works. Is it worth replacing? How balanced/effective is it? Does it default to a shared-memory setting when performed in shared memory?
    * <span style="color:blue">2022/09/20 - Based on what I can find, there's no need to change this. It's nearly as effective as `mulitiprocessing`, while also scaling to distributed systems. It is more effective than `dask` or `spark` for short jobs. Only issues may come if we change systems at some point, as `fiber` doesn't support as many cluster managers as other tools. </span>
  * **KNN** - Currently it runs and sorts all agents, when it really only needs top N. Should use a partial sort method. Is it possible to save sorted/partially-sorted lists (for how it is used, I don't think this is an option), as sorting an already partially-sorted object is faster than sorting an totally random object. 
  Is there a method for maintaing a graph of scores, and is it faster to find nearest neighbors in that setting? 
  Does this even apply, since scores will be updated at each reproduction due to current environments progressing?
  * **PATA-EC and Archiv** - When calculating the PATA-EC, all agents (archived and current) are run against all  environments (archived and current). This is silly, as archived agents never change, and environments, once created, also never change. 
    * Archived environments should save the vector of scores from archived agents, and never run those agents again. 
    * Newly created environments should also store this vector of archived scores. This may actually cover the point above.
    * When an agent/environment gets archived, it should be run against all other environments and the score vectors should be updated, so the vectors stay in the same order and then it won't need run again. 
    * Saved scored can be clipped, since everyone gets clipped, but they can't be ranked, cause current agents may change the ranking. 
    * Active Agents will always need rerun, because they are changing. However, because this is used during reproduction, and the following step is ranking agents by score, if we save these scores then they will only need run once per reproduction. 
* **Testing**:
  * PPO: Used PPO 2 from baselines, so PPO3 from stable baselines is a solid upgrade. They also did no hyperparameter tuning, so idk how valid it is to say that this *can't* solve the environments.
  * ES: They used their version of ES, but there are several other versions, some significantly more compute intensive and others not. CMA-ES is the classic, most powerful but also most resource intensive version. PEPG is a new, less compute intensive, but nearly equal in performance (on most tasks).