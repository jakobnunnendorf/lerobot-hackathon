# EOD / demining domain and prior art: evidence base for the "bomb defusal" demo idea

Researched 2026-11-20, web sources only. Purpose: critique and sharpen the idea, not sell it.

Verification tags used on every claim:
- **[F]** = I opened the cited page and read the figure there.
- **[S]** = figure comes from a search-engine extract of the cited page; the page itself was not opened (or could not be: paywall, 403/404, certificate error). Check before putting it on a slide.
- **[U]** = unverified / not found / my own inference. Do not present as fact.

## TL;DR

1. The problem is real and large: 6,279 recorded mine/ERW casualties in 2024 (90% civilians), Ukraine has ~132,000-139,000 km2 of potentially contaminated land against ~4,500 deminers, and Syria is running at about one casualty every six hours.
2. "Cut the right wire" is the weakest part of the idea. Real EOD does not select wires by colour, the international IED standard lists hand-cutting wires as a manual action to avoid, and the preferred methods are remote disruption or destruction in situ.
3. The actual bottlenecks in humanitarian clearance are survey, detection, excavation and funding, not the final manipulation step. A wire-cutting arm does not address them.
4. There is one place the technical core is genuinely on-target: EOD robots are still almost entirely teleoperated, manipulation is slow and hard to learn, and "operator assist" (click a target, robot does the fine motion) has been a research goal since at least 2008 without reaching fielded systems.
5. No SO-100/SO-101 bomb-defusal project was found. Closest prior art: an SO-101 scissors project (could not cut hard items) and a 350 kg research EOD robot where human teleoperators succeeded at one-shot wire cutting only 46-71% of the time.
6. Recommended reframe: "shared-autonomy tool use for hazardous teleoperation" (operator picks the target, learned policy executes the cut), demonstrated on a neutral prop, with EOD / battery disassembly / live-line work as the motivating applications rather than a movie-style bomb.

---

## 1. Real problem size

### Global
- 6,279 people killed or injured by landmines and explosive remnants of war in 2024 (1,945 killed), in 52 countries and areas; highest annual figure since 2020. Myanmar 2,029, Syria 1,015. Source: Landmine Monitor 2025 via HRW, https://www.hrw.org/news/2025/12/01/landmines-action-needed-to-reinforce-ban **[F]**
- Civilians were 90% of casualties; almost half were children. https://news.un.org/en/story/2025/12/1166476 **[F]**
- At least 57 states and other areas are contaminated; seven "massively" (Afghanistan, Bosnia and Herzegovina, Cambodia, Ethiopia, Iraq, Turkiye, Ukraine). Same UN source **[F]**
- Clearance in 2024: two different numbers, different methods. Landmine Monitor (via HRW): 1,115 km2 "cleared" and 105,640 antipersonnel mines destroyed (the Monitor's own wording appears to be land "released", which includes survey) **[F for HRW wording, U for the cleared/released distinction]**. Mine Action Review: 175.39 km2 of mined land cleared, 131,190 AP mines destroyed, clearance down 26% on 2023. https://maginternational.org/whats-happening/2025-clearing-the-mines-report-out-now/ **[F]**
- Funding is shrinking: international mine action funding fell 5% in 2024; the US remained the largest donor but its contribution fell by more than one-third (HRW link above **[F]**). The 2025 US funding freeze shut programmes in Afghanistan, Iraq, Yemen, Colombia, Tajikistan and Zimbabwe (UN News link above **[F]**); Norwegian People's Aid reportedly laid off 1,700 workers in 12 countries, https://www.kyivpost.com/amp/post/46955 **[S]**
- Cost asymmetry (old, widely repeated UN estimate): $3-75 to produce a mine, $300-1,000 to remove one. Wikipedia "Deminer" citing the UN, https://en.wikipedia.com/wiki/Deminer **[S]**. Treat as indicative only.
- Global number of deminers / EOD technicians: **not found [U]**.

### Ukraine
- Potentially contaminated area: more than 132,000 km2 per state-run Demine Ukraine, with nearly 42,000 km2 made safe (Reuters, 30 Apr 2026, republished at https://www.thestar.com.my/news/world/2026/04/30/man-machine-and-ai-combine-in-battle-to-clear-ukraine-of-mines **[F]**). Ministry of Defence figure: 139,000 km2 including 14,000 km2 of water, https://mod.gov.ua/en/news/over-the-year-the-scope-of-humanitarian-demining-in-ukraine-has-increased-by-50 **[S]**
- Cost: $29.8 billion (RDNA4, World Bank / Government of Ukraine / EU / UN, Feb 2025), down ~$4.8 billion from the previous estimate. https://www.kmu.gov.ua/en/news/potreby-ukrainy-u-rozminuvanni-skorotylysia-na-5-mlrd-zavdiaky-prohresu-v-ochyshchenni-zemel-zvit-rdna4 **[S]**
- Workforce: about 4,500 trained specialists against a need for "tens of thousands" (Stockholm School of Economics SITE policy brief, 1 Dec 2025, https://www.hhs.se/news/site-publications/2025/ukraine-demining-efforts **[F]**). Tony Blair Institute (Sept 2024): 10,000 deminers needed, a third already in the field, https://institute.global/insights/geopolitics-and-security/the-urgent-need-to-demine-ukraine **[S]** (page returned HTTP 429).
- 132 certified mine action operators (organisations) at end-2025, up 60 in the year. https://mod.gov.ua/en/news/in-2025-the-ministry-of-defence-of-ukraine-s-mine-clearance-units-cleared-over-66-200-hectares-of-liberated-territories **[S]**
- Time: "Ukraine will not be demined in less than 10 years" (HALO, Reuters link above **[F]**). The often-quoted "757 years" is a 2023 GLOBSEC extrapolation from ~500 teams and 174,000 km2 and is out of date, https://www.globsec.org/sites/default/files/2023-04/Demining%20in%20Ukraine%20report%20ver5%20web.pdf **[S]**. Do not use it without that caveat.
- Economic cost: more than $11 billion of GDP per year (TBI / Ministry of Economy, 2024), https://www.kmu.gov.ua/en/news/minekonomiky-ukraina-shchoroku-vtrachaie-112-miliarda-dolariv-cherez-zaminovani-terytorii **[S]**
- Clearance personnel casualties: 21 State Emergency Service sappers killed during demining since the full-scale invasion, with 77 or 87 injured depending on the report (date of the statement not verified, roughly 2024). https://mvs.gov.ua/en/news/igor-klimenko-pid-cas-rozminuvan-z-pocatku-povnomasstabnoyi-viini-zaginuv-21-pracivnik-dsns-shhe-87-otrimali-poranennia and https://www.ukrinform.net/rubric-ato/3848683-demining-ukraine-21-sappers-die-on-job.html **[S]**

### Syria
- At least 2,375 people killed or injured by unexploded ordnance since December 2024, including more than 850 children; "one person becoming a casualty every six hours" (HALO); almost 700 casualties in 2026 to date; 20-30 years to clear; shortages of "resources, trained personnel and advanced equipment". AFP via Al-Monitor, 1 Sep 2026, https://al-monitor.com/originals/2026/09/syria-uxo-causing-death-injury-every-six-hours-charity **[F]**
- HALO had funding for only ~40 deminers when the regime fell, ~120 by mid-2025, 250+ staff later. https://www.jmu.edu/news/cisr/2025/01/15-syria.shtml and https://reliefweb.int/report/syrian-arab-republic/syrian-landmine-casualties-set-spiral-school-year-ends **[S]**
- No nationwide survey exists; total contaminated area unknown (Al-Monitor link above **[F]**).

### Gaza
- More than 1,000 people killed by remnants of war since October 2023 ("certainly a severe under-estimate"), half of known casualties children; more than 1,000 items identified; roughly one item every 600 m; clearance cost $541 million in a best case; "We have barely scratched the surface". UNMAS briefing via Al-Monitor, 24 Apr 2026, https://www.al-monitor.com/originals/2026/04/unexploded-bombs-littering-gaza-threaten-recovery-decades-un-warns **[F]**
- Around 61 million tonnes of debris; Humanity & Inclusion estimate of 20-30 years to clear the surface. https://www.gulftoday.ae/news/2025/10/25/clearing-gazas-surface-of-bombs-will-take-up-to-30-years-aid-group **[S]**

### Iraq
- About 2,100 km2 contaminated (ICRC, April 2025), https://www.icrc.org/en/news-release/landmines-and-explosive-remnants-cast-long-shadow-over-iraq-amid-recovery-efforts **[S]**. Other sources give 1,636-2,583 km2; definitions differ.

### Trained EOD operators (military / police)
- United States: about 3,100 public safety bomb technicians in 467 bomb squads, ~200 new technicians certified per year (FBI, 2016 figures; a decade old). https://www.fbi.gov/news/stories/fbi-takes-lead-role-in-training-nations-public-safety-bomb-technicians **[S]**
- Training is long and lossy: Naval School EOD attrition reported at 33-42% in Air Force articles (2011-era data), https://www.af.mil/News/Article-Display/Article/110667/eod-prelim-helps-tech-school-attrition-rate/ **[S]**. Current rate **[U]**.
- Deminer accident rate: no reliable global figure exists. One study suggested about one severe accident per 32 field deminers per year, and the article itself warns the ratio may be misleading. CISR Journal 23.1, https://issuu.com/cisr-journal/docs/journal-23.1-all/s/96036 **[S]**
- Historic EOD losses: 23 British ATOs killed in Northern Ireland; over 400 Wheelbarrow robots destroyed 1972-78, "each loss often meant an operator's life was saved"; three experts with 20+ years' experience killed by a WWII bomb in Gottingen in 2010. https://en.wikipedia.org/wiki/Bomb_disposal **[F]**

---

## 2. How real EOD / demining works, and how realistic "cut the right wire" is

- Colour is meaningless. Former USAF EOD officer Brian Castner: "the wire colors have absolutely no correspondence to absolutely anything" and "You actually don't cut any of them." Also: "You send in the robot first. If that robot blows up, you send another." Business Insider interview syndicated 29 Oct 2025, https://www.yahoo.com/news/articles/former-air-force-bomb-disposal-173601726.html **[F]**
- Doctrine is remote-first: "The role of modern bomb disposal operators is to accomplish their task as remotely as possible." https://en.wikipedia.org/wiki/Bomb_disposal **[F]**
- The standard tool is a disruptor, not a cutter: it fires water at high velocity to disable electronics and separate components before the device can function. https://bostondynamics.com/blog/what-is-a-disrupter/ **[F]**. "Pigstick fires a jet of water driven by a propellant charge to disrupt the circuitry of a bomb" (Wikipedia link above **[F]**).
- Humanitarian IED disposal standard (IMAS 09.31, quoted in the GICHD IED Clearance Good Practice Guide): Principle 1, "Manual neutralisation techniques should not be conducted. Remote (if available) and semi-remote actions should be conducted"; listed examples of manual actions to avoid include "Cutting wires or detonating cord by hand." Principle 2: destruction in situ with a donor charge "is the preferred method of disposal" when feasible. https://epubs.gichd.org/view/351700/241 **[S]** (site had an expired certificate; landing page https://gichd.org/our-response/mine-action-standards/improvised-explosive-device-clearance-good-practice-guide).
- Reading that standard carefully cuts both ways: the prohibition is on cutting *by hand*. Doing the same action remotely is exactly what the standard wants. That is the legitimate opening for a robot. **[U, my inference]**
- Semi-remote work (hook and line) is a core humanitarian IEDD skill (same GICHD guide **[S]**).
- Where fine manipulation genuinely matters:
  - Opening bags and containers, unscrewing pipe-bomb end caps: L3Harris describes its dual-arm T7 payload as enabling "unzipping backpacks and unscrewing end-caps from pipe bomb IEDs". https://www.l3harris.com/newsroom/editorial/2025/02/l3harris-demonstrates-new-robot-dual-arm-capability-uk-ministry-defence **[F]**
  - Wire cutting is used as a benchmark task in EOD robotics research: the FC-EODR paper calls it "the typical scene of wire-cutting during the EOD operation". https://pmc.ncbi.nlm.nih.gov/articles/PMC9944835/ **[F]**
  - Opening doors, placing and aiming disruptors, X-ray placement (UK Dstl trial, section 3).
  - Fuze removal on conventional ordnance: historically central and extremely dangerous; anti-handling fuzes "were specifically designed to kill bomb disposal personnel" (Wikipedia link above **[F]**). In humanitarian demining the default is to destroy in place instead.
  - Tripwire handling and careful excavation in manual demining: the most common accident is initiating a detected mine while exposing it with hand tools. https://www.gichd.org/fileadmin/uploads/gichd/Publications/Manual_Mine_Clearance_FullBook.pdf **[S]**. Specific tripwire-cutting procedure **[U]**.
- Bottom line: selective wire cutting is mostly a film trope. Component separation by remote means is real, but it is done with disruptors, hook-and-line and teleoperated manipulators, and the decision of what to cut comes from X-ray and diagnosis by a trained human, not from colour.

---

## 3. State of the art in EOD robots and the role of AI

### Fielded systems are teleoperated
- TALON: more than 4,000 delivered since 2000, https://www.army-technology.com/?p=21105 **[S]**. US RS-JPO fielded over 6,000 ground robots since 2004 (all types), https://www.army.mil/article/197503/how_many_robots_does_it_take **[S]**
- UK Project STARTER: 122 L3Harris T7 robots (56 for 55.3 million GBP, then a 36 million GBP option), delivered by 2021, with haptic feedback. https://www.l3harris.com/newsroom/trade-release/2021/06/l3harris-technologies-delivers-final-explosive-ordnance-disposal **[S]**
- tEODor, MTRS Inc II / Centaur, CRS-H fleet numbers: **not found [U]**.

### Manipulation autonomy has been "coming" for 18 years
- 2008, Naval EOD Technology Division: "Robotic manipulators used on current EOD robotic platforms exhibit very few autonomous capabilities. This lack of autonomy forces the operator to completely control manipulator movements... While autonomous manipulation has also been researched, this technology has yet to appear in fielded EOD robotic systems." They proposed point-and-click moves, saved positions and macros. https://remotesensing.spiedigitallibrary.org/conference-proceedings-of-spie/6962/69620M/Low-cost-semi-autonomous-manipulation-technique-for-explosive-ordnance-disposal/10.1117/12.782082.full **[S, abstract only]**
- CMU NREC (Navy-funded) operator assist: click a target and the robot plans and drives there, to "make EOD robots easier to teleoperate, lower operator workloads". No quantitative results published on the page. https://www.nrec.ri.cmu.edu/solutions/defense/eod-robot-operator-assist **[F]**. NREC manipulator coordination: click on an object and the robot reaches and grabs it, https://www.ri.cmu.edu/project/manipulator-coordination **[S]**
- US AEODRS programme roadmap (2017) listed fly-the-end-effector, autonomous grasping and autonomous tool swapping as planned capabilities. https://ndia.dtic.mil/wp-content/uploads/2017/groundrobot/Ryan.pdf **[S]**
- Autonomous Solutions: semi-autonomy "still requires human oversight and intervention, but at a much lower rate". https://www.sci.utah.edu/~jedwards/research/ManipulatorAutonomy.pdf **[S]**

### Dual-arm
- L3Harris T7 + HDT Adroit dual-arm payload shown to UK MOD (Nov 2024): one high-strength arm, one high-dexterity arm, haptic controller. A human-factors study with the US Army Research Lab reportedly found a 20% reduction in operator workload and a 20% reduction in task time (vendor-reported). L3Harris link in section 2 **[F]**
- FC-EODR (Biomimetics, 2023): 350 kg dual-arm EOD robot. Ten novice operators doing one-shot wire cutting by teleoperation: success rate 0.46 with a VR controller, 0.71 with the proposed 6D-mouse plus assistance; 107.9 s vs 70.1 s average. The paper says joint-space joystick control "is not intuitive, an operator needs much learning and training". https://pmc.ncbi.nlm.nih.gov/articles/PMC9944835/ **[F]**
- Bimanual haptic telepresence for EOD (TU Munich), https://www.ce.cit.tum.de/fileadmin/w00cgn/lsr/gs/pres_puc/Bimanual_Haptic_Telepresent_Control__and_its_Application_to_Deactivation_of_Mines.pdf **[S]**

### Recent AI / autonomy trials
- UK Dstl, Feb 2025: four-day trial with L3Harris, Marlborough Communications and AeroVironment. Robots opened and closed doors autonomously, climbed stairs, inspected IEDs and fired disruptors; drones used AI threat and people detection. Stated aim: "Reducing the need to put a bomb disposal operator in harm's way". Only door operation and drone detection are described as autonomous. https://www.gov.uk/government/news/new-robots-lead-the-way-in-bomb-disposal-innovation **[F]**
- Ukraine: HALO uses AI on drone imagery at about 70% accuracy, and remote excavators operated with VR glasses and a joystick (Reuters link in section 1 **[F]**). HALO's DMR-7 remotely operated vehicle excavates a patch in about 5 minutes versus up to 45 by hand, https://thedefensepost.com/2026/10/15/drone-warfare-ukraine-explosive-hazards/amp/ **[F]**
- So in humanitarian clearance, AI is being applied to detection and survey, and remote machines to excavation. Not to render-safe manipulation.

### Imitation learning / VLA in hazardous manipulation
- No published imitation-learning or VLA work specifically for EOD or demining manipulation was found **[U: absence of evidence from a few searches, not proof]**. This is either an open niche or a sign that the field does not see it as the bottleneck.
- Adjacent: KAIST (2025), "Teaching Robots to Handle Nuclear Waste: A Teleoperation-Based Learning Approach", learns from recorded teleoperation to cut operator involvement, https://arxiv.org/abs/2504.01405v1 **[S]**. Birmingham nuclear pilot study found semi-autonomous assistance improved novice operators' performance, https://research.birmingham.ac.uk/en/publications/towards-advanced-robotic-manipulation-for-nuclear-decommissioning **[S]**
- Surgical precedent: Johns Hopkins SRT-H (Science Robotics, July 2025) completed all 17 steps of a gallbladder-removal phase in 8 of 8 trials by imitation learning with a language-conditioned high-level planner, ~41% slower than an expert. https://hub.jhu.edu/2025/07/09/robot-performs-first-realistic-surgery-without-human-help **[S]**

### Why robots are not already everywhere in demining
- Long-standing objections: cost, sensing limits, variety of mines and terrain, and the requirement for near-100% reliability so that cleared land still needs manual verification. Trevelyan 1997, https://www.emerald.com/insight/content/doi/10.1108/01439919710165725/full/html **[S]** (old literature).

### Credible near-term role for AI
Assist, not replace: (a) detection and prioritisation from imagery, (b) operator assist for teleoperated manipulators so that a less experienced operator can do fine tasks faster, with a human approving every consequential action, (c) training simulators. Nothing found supports a claim that autonomous render-safe is near.

---

## 4. Prior art in the hobby / hackathon space

- No SO-100 / SO-101 / LeRobot "bomb defusal" or wire-cutting project was found in several searches **[U: not exhaustive; Discord and X were not searchable]**. The novelty claim is plausible but unproven.
- Closest hobby prior art: "Creative Tool Use" (Jared Bailey, Nov 2025). Scissors on a custom SO-101 end effector, ACT policies per subtask, a state machine with two YOLO classifiers verifying progress between subtasks. Findings: the SO-101 "couldn't cut harder foods reliably" so softer items were used; the snap-fit mount failed at small scale and was fixed with superglue and tape; the robot sometimes made repeated cuts; data quality mattered more than model choice. No episode counts or success rates given. https://jaredlbailey.github.io/2025/11/01/Imitation-Learning.html **[F]**
- Research prior art for learned cutting: ScissorBot (paper cutting with scissors via simulation, imitation and sim2real), https://arxiv.org/abs/2409.13966 **[S]**. Diffusion-policy prying for battery disassembly, 96% success, https://arxiv.org/pdf/2503.03998 **[S]**
- Bomb-themed events: UK Defence AI Centre hackathon (Nov 2023), 40 programmers on AI responses to bomb threats, https://www.techtimes.com/articles/298889/20231120/uk-governments-defense-ai-center-leads-innovative-hackathon-ai-powered-responses-bomb-threats.htm **[S]**. A 2017 DEF CON forum thread proposed a robot contest to cut a fake bomb's wire; no evidence it ran, https://forum.defcon.org/node/226029 **[S]**
- LeRobot Worldwide Hackathon (June 2025) top-30 list exists only as video filenames: 1 LeChef, 2 ez_2_ai, 3 ML6, 4 Electrify, 5 data-surgeon, 11 Scewdriver, and others. https://huggingface.co/api/datasets/maringetxway/all-winners/tree/main **[F]**. What these projects did is **[U]**; watch the videos at https://huggingface.co/spaces/LeRobot-worldwide-hackathon/all-winners before claiming novelty.

### What made winning projects stand out (from write-ups that could be read)
- Seeed Embodied AI Hackathon, Oct 2025 (https://seeed.cc/post/2025-embodied-ai-hackathon-recap **[F]**):
  - US champion (matcha making, bimanual, GR00T N1.5): gave each subtask a distinct start and end state to work around the model's lack of memory.
  - US runner-up (sugar sprinkling, SmolVLA, 170 episodes): split a long-horizon task into four focused models.
  - China champion (desk tidying, GR00T N1.5, 300 episodes, 90% real / 10% sim): stable, generalising behaviour.
  - Folding entries: full pipeline in a day and a half, recovery from unseen errors.
- London, LeRobot Worldwide Hackathon: "Spice Terminator" (plate wiping, 75 episodes, GR00T N1.5 and SmolVLA) tied for Most Novel Solution. The author credits clear roles, a clear testable problem, and showing robustness live (moving the plate mid-task). Demo-day failures were real: grabbed the wrong object, got stuck. "Optimizing the demo matters as much as optimizing the codebase." https://darthgera123.github.io/blogs/hackathon/robot **[F]**
- London: "Episode-124" (waste sorting on a moving conveyor, 124 episodes, ACT then SmolVLA then GR00T N1.5) took the Technical Experts Award and 2nd overall. https://om-kulkarni.github.io/projects/waste-segregation-hackathon **[F]**
- Mountain View top-five: voice-commanded tool-handing arm for mechanics; SmolVLA; a pitch video; a concrete user story. https://robotics.umich.edu/news/2025/hackathon-qa-brouze **[F]**
- Pattern: 75-300 episodes, long tasks decomposed into verifiable subtasks, a robustness test performed live in front of judges, a concrete user, and a rehearsed demo. Official LeRobot judging rubric **not found [U]**.

---

## 5. Risks, objections, and better framings

### Objections a judge could raise
1. **Realism.** EOD technicians do not pick wires by colour (Castner **[F]**) and the humanitarian standard says not to hand-cut wires and to prefer destruction in situ (IMAS 09.31 **[S]**). A judge with any defence or mine-action background will say this in the first question.
2. **Wrong bottleneck.** The scarce things are survey, detection, excavation capacity and money (funding down, US cut by a third **[F]**; Syria has no national survey **[F]**; AI in the field is at ~70% detection accuracy **[F]**). The pitch line "more hazards than operators, so autonomy scales disposal" jumps from a true premise to a conclusion the evidence does not support.
3. **"Why not a colour classifier plus scripted motion?"** Fair. Colour selection is a few lines of OpenCV. LeRobot's ACT is not language-conditioned (the docs say `--task` "can be skipped for ACT", https://huggingface.co/docs/lerobot/en/act **[F]**), so "cut the green one" is not something a single ACT policy gives for free. The closest prior project used exactly a classifier plus state machine around learned subtasks (Bailey **[F]**). The team needs a crisp answer for what is learned and why it has to be: approaching a thin deformable wire at varying pose, getting the blade around it, bimanual coordination, recovery after a miss. **[U, my assessment]**
4. **Reliability.** Purpose-built EOD hardware with human teleoperators got 46-71% on one-shot wire cutting (FC-EODR **[F]**). A weekend policy on hobby servos will be lower, and in this domain a miss is the whole story. Claiming "defusal" invites the question "what is your failure rate and what happens on failure?"
5. **Hardware.** The SO-101 could not cut harder foods reliably (Bailey **[F]**). Reported payload 150-500 g depending on source and about 7.5 mm deflection at 300 g, https://www.roboticscenter.ai/hardware/so-101/specs and https://raw.githubusercontent.com/roboninecom/SO-ARM-102/main/README.md **[S]**. Whether the gripper can drive cutters through real wire is **[U]**: test on day one with the thinnest wire, string or a pre-scored prop.
6. **Blade safety.** An exposed blade on an arm that will be teleoperated by tired people next to spectators is an avoidable hazard. No hackathon blade rule was found **[U]**; ask the organisers before building. Enclosed-jaw flush cutters or scissors are safer than a fixed knife and are what the prior project used.
7. **Optics and ethics.** A prop with wires and a timer at a public venue can be mistaken for the real thing; a San Diego school was evacuated in 2015 when a student's science project was mistaken for a bomb, https://www.foxnews.com/story/california-middle-school-evacuated-after-science-project-mistaken-for-bomb.print **[S]**. EOD manipulation research is largely military-funded (Navy EOD Technology Division, Dstl; section 3), and the mine-ban regime is politically raw right now (five European states withdrawing, Ukraine suspending obligations; HRW **[F]**). Presenting a toy as a solution for Gaza or Ukraine can read as trivialising. Dual-use concern is modest for a wire cutter but will be raised.
8. **Human in the loop.** Every credible source frames autonomy as reducing operator burden with human oversight (Autonomous Solutions **[S]**, Dstl **[F]**). "Autonomous defusal" is the least defensible wording available.

### Better framings of the same technical core
| Framing | Why stronger | Why weaker |
|---|---|---|
| **Shared-autonomy operator assist for hazardous teleoperation** ("operator points, robot cuts") | Matches an 18-year-old documented gap (NAVEOD 2008 **[S]**, NREC **[F]**, AEODRS **[S]**); has numbers to beat (46-71% novice teleop success **[F]**, 20% workload reduction **[F]**); keeps the human deciding, which answers the ethics and reliability objections; lets the team show an A/B: teleop time vs assisted time. | Still EOD-adjacent, so needs a neutral prop; requires an honest comparison experiment in two days. |
| **EV battery / e-waste disassembly** (cut the right cable, second arm holds the part) | Disassembly is still largely manual because of design variability (https://arxiv.org/abs/2603.18520 **[S]**); e-waste was 62 million tonnes in 2022 with ~22% formally recycled (https://ewastemonitor.info/the-global-e-waste-monitor-2024/ **[S]**); active robot-learning literature; no optics problem; "which cable" is a real perception question. | Less dramatic; injury statistics for manual disassembly not found **[U]**; real packs need far more force than an SO-101 has. |
| **Live-line electrical work** | Here "cut / strip / connect the correct wire" is literally the job; autonomous 10 kV live-working robots exist and reach near-human efficiency (https://doaj.org/article/468b81b7f7464c30b87e4320691e5c0f **[S]**); "fully autonomous mode is uncontrollable and teleoperation has a high failure rate" motivates shared control (https://emerald.com/insight/content/doi/10.1108/IR-07-2023-0154 **[S]**). | Lineworker fatality and deployment numbers not found **[U]**; existing industrial robots already do it, so novelty is low. |
| **Humanitarian demining assistant** | Strongest emotional pull and the best problem-size numbers (section 1). | Weakest technical fit: standards favour destroy-in-situ, the bottleneck is detection, no evidence that fine manipulation is the limiting step; highest risk of seeming to trivialise. |
| **Surgical-style precision cutting** | Strong imitation-learning precedent (SRT-H, 17 steps, 8/8 **[S]**). | SO-101 precision is nowhere near surgical; invites an overclaim objection. |
| **Game framing** ("Keep Talking"-style: a human expert gives verbal instructions, a language-conditioned policy executes) | Honest about being a benchmark for language-conditioned bimanual tool use; fun; sidesteps realism claims entirely. | No social-impact story; this is my suggestion with no external validation **[U]**. |

---

## Strongest pitch arguments (each with source)

1. The human cost is current and rising: 6,279 casualties in 2024, 90% civilian, almost half children. https://www.hrw.org/news/2025/12/01/landmines-action-needed-to-reinforce-ban ; https://news.un.org/en/story/2025/12/1166476
2. Capacity is far below need: Ukraine ~4,500 specialists vs "tens of thousands" needed for 132,000+ km2; Syria one casualty every six hours with stated shortages of trained personnel. https://www.hhs.se/news/site-publications/2025/ukraine-demining-efforts ; https://al-monitor.com/originals/2026/09/syria-uxo-causing-death-injury-every-six-hours-charity
3. Doctrine already wants the human away from the device: remote-first, "send in the robot first", and IMAS 09.31 forbids manual neutralisation. https://www.yahoo.com/news/articles/former-air-force-bomb-disposal-173601726.html ; https://epubs.gichd.org/view/351700/241
4. Manipulation autonomy is a documented, unfilled gap: "has yet to appear in fielded EOD robotic systems" (2008), still on roadmaps in 2017, and 2025 trials only automate door opening. https://remotesensing.spiedigitallibrary.org/conference-proceedings-of-spie/6962/69620M/Low-cost-semi-autonomous-manipulation-technique-for-explosive-ordnance-disposal/10.1117/12.782082.full ; https://www.gov.uk/government/news/new-robots-lead-the-way-in-bomb-disposal-innovation
5. Teleoperated fine manipulation is measurably hard and trainable-operator-limited: 46-71% novice success on one-shot wire cutting; joystick control "needs much learning and training". https://pmc.ncbi.nlm.nih.gov/articles/PMC9944835/
6. Dual-arm is where the industry is heading (hold with one, work with the other), and better interfaces already show ~20% workload and time reductions. https://www.l3harris.com/newsroom/editorial/2025/02/l3harris-demonstrates-new-robot-dual-arm-capability-uk-ministry-defence
7. Learning from teleoperation demonstrations is proven in adjacent high-stakes domains (surgery, nuclear waste). https://hub.jhu.edu/2025/07/09/robot-performs-first-realistic-surgery-without-human-help ; https://arxiv.org/abs/2504.01405v1
8. Apparent novelty in the LeRobot community: no comparable project found (unverified; check the winners' videos). https://huggingface.co/spaces/LeRobot-worldwide-hackathon/all-winners

## Strongest objections (each with source)

1. Colour-coded wire cutting is fiction: "the wire colors have absolutely no correspondence to absolutely anything... You actually don't cut any of them." https://www.yahoo.com/news/articles/former-air-force-bomb-disposal-173601726.html
2. Standards point the other way: disruptors and destruction in situ are preferred; wire cutting by hand is a listed action to avoid. https://epubs.gichd.org/view/351700/241 ; https://bostondynamics.com/blog/what-is-a-disrupter/
3. It does not address the real bottleneck (survey, detection, funding). https://www.thestar.com.my/news/world/2026/04/30/man-machine-and-ai-combine-in-battle-to-clear-ukraine-of-mines ; https://www.hrw.org/news/2025/12/01/landmines-action-needed-to-reinforce-ban
4. The perception part is trivial and ACT gives no language conditioning, so "CV picks the wire" adds little that is learned. https://huggingface.co/docs/lerobot/en/act ; https://jaredlbailey.github.io/2025/11/01/Imitation-Learning.html
5. Reliability will be far below what the word "defusal" implies; even dedicated hardware with humans in the loop reaches 46-71%. https://pmc.ncbi.nlm.nih.gov/articles/PMC9944835/
6. The SO-101 struggles to cut anything hard and flexes under load. https://jaredlbailey.github.io/2025/11/01/Imitation-Learning.html ; https://www.roboticscenter.ai/hardware/so-101/specs
7. Demining robots have a 25-year history of failing on cost and the near-100% reliability requirement. https://www.emerald.com/insight/content/doi/10.1108/01439919710165725/full/html
8. Optics: bomb-like props cause real security incidents, and the subject is politically sensitive. https://www.foxnews.com/story/california-middle-school-evacuated-after-science-project-mistaken-for-bomb.print ; https://www.hrw.org/news/2025/12/01/landmines-action-needed-to-reinforce-ban
9. Blade safety at the venue: no rule found either way; unresolved until the organisers are asked. (unverified)

## Gaps worth closing before committing
- Watch the top-30 LeRobot Worldwide Hackathon videos to confirm nobody did cutting or tool use this way.
- Find this event's judging rubric and any tool / blade restrictions.
- Bench test: can the SO-101 gripper close cutters through the chosen wire, repeatably, at the poses the policy will use?
- If keeping any EOD framing, get five minutes with someone who has done the job; one sentence from them is worth more than every statistic above.
