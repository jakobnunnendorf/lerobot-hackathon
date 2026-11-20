# Ball picking, stirring and handover: evidence review

Researched 2026-11-20 from public web sources only. Every claim carries a source link. Claims marked **[unverified]** or **[estimate]** are collected again in the last section. Success rates quoted from hobby write-ups usually rest on 5 to 10 trials, so treat them as anecdotes, not measurements.

## TL;DR

1. Decomposition is the better-supported choice for this task, but the evidence is indirect: I found no public write-up of an SO-100/SO-101 team running separate ACT skills under a state machine, so nobody has shown that exact design working or failing.
2. The strongest version of the plan is narrower than "one ACT per action": learn only PICK (and perhaps GIVE), and script or replay MIX, PUT DOWN and SHAKE. Long single-policy tasks on this hardware have a poor record (about 10% success with 175 episodes in one hackathon report).
3. Use a classical colour detector inside a fixed bowl mask as the state-machine sensor, with a small fine-tuned YOLO as the fallback. The detector should gate skills, not feed the policy.
4. The physical risks are larger than the ML risks: balls cluster at the bottom of the bowl, a parallel jaw needs clearance on both sides, the arm hides the bowl from the top camera, and foil reflects the blue ball.
5. MediaPipe works on Python 3.12 on Windows and Linux, but only through the Tasks API (the old `mp.solutions` is gone), and it pulls in a second OpenCV package that clashes with LeRobot's.
6. Pitch it as mechanical search plus human handover. No earlier LeRobot hackathon project I could find combines the two, but the 30-entry winners list has no descriptions, so novelty is likely, not proven.

---

## 1. Decomposed skills plus a state machine versus one end-to-end policy

### What the evidence says

**Long tasks lose success at every stage, even with a good policy.**
- The original ACT paper trained on 50 demonstrations per real task (100 for Thread Velcro). Final success was 96% on Slot Battery and 88% on Slide Ziploc, but 64% on Prep Tape and 20% on Thread Velcro, where the text reports 92% at the first stage. On simulated Bimanual Insertion with human data the stages were 76%, 66%, 20%. The authors attribute this to errors compounding from one stage to the next. Source: https://ar5iv.labs.arxiv.org/html/2304.13705
- A hackathon team trained an SO-100 to put 1 to 3 pens into a holder, one long task, with 175 episodes (91 on day one, 84 on day two). ACT trained for 100,000 steps reached "a success rate of about 10%"; SmolVLA, fine-tuned for only 2,500 steps, did worse. Source: https://bissmella.github.io/files/PensToStand.pdf
- A benchmark of ACT against larger models reports "near-miss grasps, premature releases, and long-horizon state drift" as recurring failures, with ACT the most stable in-distribution and the weakest out of distribution. Source: https://arxiv.org/abs/2511.11298

**Teams that split reasoning from motion got a working demo in 48 hours.**
- LeX-O (tic-tac-toe, LeRobot Worldwide Hackathon): an LLM chose the cell and a single ACT, modified to take a task ID, did the pick-and-place. They recorded 10 demonstrations per grid cell, trained about 160 minutes on an RTX 5090, and say a 15,000-step checkpoint was as good as 25,000. They abandoned their first idea, shape sorting, because "even basic pick-and-place is hard". Source: https://aadarshram.github.io/2025-07-29-Le-X-O
- A second tic-tac-toe team trained one ACT on 50 demonstrations that also had to learn the game rules, and wrote afterwards that a classical solver feeding a grasping ACT "might have been more" effective. I could only read this through a search snippet; the page itself returned 403. **[unverified]** Source: https://www.hackster.io/team-bubble-bot/tic-your-tic-tac-toe-playing-robot-with-embodied-ai-02f91e

**Learning only the contact-rich part is far more data-efficient (research result, not hobby hardware).**
- HITL-TAMP lets a planner do the free-space motion and a learned policy do only the contact-rich segments. In simulation it reached 100% against 28% for conventionally collected data on Coffee, and 84% against 0% on Square; agents with 75%+ success were trained from 10 minutes of non-expert teleoperation. Sources: https://hitltamp.github.io and https://arxiv.org/abs/2310.16014

**What speaks for end-to-end.**
- The hackathon winners I could read used one fine-tuned vision-language-action model on one continuous task: a waste-sorting arm (GR00T N1.5, 124 episodes, 2nd place London) and a plate-wiping arm (GR00T N1.5 and SmolVLA, 75 episodes, Most Novel Solution London). The sorting team tried ACT from scratch and SmolVLA first (SmolVLA "faced prediction challenges"; no ACT result is given) and chose GR00T as the final solution. Sources: https://om-kulkarni.github.io/projects/waste-segregation-hackathon and https://darthgera123.github.io/blogs/hackathon/robot
- A 2026 benchmark on the SO-101 finds that pretrained VLA policies "generally outperform" ACT, with execution instability the dominant failure. Source: https://arxiv.org/abs/2606.08881
- These do not transfer cleanly to this team. GR00T and pi0-class models need about 24 to 40 GB of VRAM to train and SmolVLA 10 to 16 GB, against 2 to 6 GB for ACT, so the 8 GB laptop can only train ACT. Source: https://huggingface.co/docs/lerobot/hardware_guide
- Both winning tasks were single cohesive behaviours. This task has two branches (ball visible or not, hand present or not) and five behaviours, so 100 to 200 episodes would be split five ways inside one policy.

### How handoffs fail, and what is known to help

**Start-pose mismatch.** Chaining fails when a skill starts from a state it never saw in training. The standard remedies are to widen each skill's set of starting states, or to constrain where the previous skill ends. Source: https://arxiv.org/abs/2111.07999 (Lee et al., CoRL 2021). On hobby hardware the cheap version is: every skill begins and ends at one fixed home pose, and a scripted move returns the arm there between skills. The current LeRobot docs expose this: `RolloutController.reset()` is documented as "stop movement, return home, stay warm", and the episodic strategy has `--strategy.reset_to_initial_position`. Source: https://huggingface.co/docs/lerobot/inference

**The policy does not know it is done.** `lerobot-rollout` ends a run by `--duration` only; the documentation describes no success detection. Source: https://huggingface.co/docs/lerobot/inference. Research answers are an external check: compare the current frame with stored success frames, or train a separate termination predictor. Sources: https://arxiv.org/abs/2602.01811 and https://arxiv.org/abs/1810.11043. Mechanical-search work simply uses hard limits (see section 4).

**Switching checkpoints.** The documented CLI and `RolloutController` take one `--policy.path`. I found no documentation or issue describing two policies in one process, so plan on a custom loop that loads each ACT once at start-up and resets the policy's action queue when switching. **[unverified]** that several rollout contexts can share one robot connection. Source: https://huggingface.co/docs/lerobot/inference

**Speed matters as much as the checkpoint.** One SO-101 ACT policy scored 9/10 at 30 Hz on a CUDA machine and 4/10 at about 15 Hz on a Mac, which the author attributes to the frame-rate mismatch; turning on `--display_data` cut the loop from 30 Hz to 9.8 Hz. Source: https://dev.classmethod.jp/en/articles/lerobot-so-arm101-act-training-eval/. Anything else running on the laptop during a skill (detector, hand tracker, viewer) competes with the policy.

### Verdict

Decomposition is supported by: stage-wise success loss in single policies, one direct failure of a long single-policy SO-100 task, two hackathon teams pointing to hybrid designs, and the HITL-TAMP result. It is not supported by a direct like-for-like comparison on an SO-101, and the handoff is where it will break. The mitigations are all cheap and all scripted.

---

## 2. Picking one coloured ball among distractors out of a bowl

**Episode counts.** The LeRobot guide recommends "at least 50 episodes, with 10 episodes per location", fixed cameras, consistent grasping, and warns "Avoid adding too much variation too quickly". Source: https://huggingface.co/docs/lerobot/il_robots. Community numbers for a single pick-and-place: 30 fixed-start episodes gave 5/5; adding 20 varied-start episodes gave 3/7 on varied starts (each start tested once). Source: https://openelab.io/blogs/learn/so-arm101-pick-and-place-lerobot-act-case-study. Sixty episodes over 4 start locations gave ACT 100% where Diffusion Policy got 25 to 50%. Source: https://flaviogheri.github.io/projects/imitation_learning/. A forum thread on ACT with 50 episodes concludes that coverage beats count: misses cluster where demonstrations are sparse, and one LeRobot issue found that going from 50+ to 120+ episodes did not fix unseen position combinations. Source: https://discuss.huggingface.co/t/what-is-the-expected-success-rate-for-a-act-trained-model-using-50-episodes/178517

**ACT may not look at the ball at all.** LeX-O's attention analysis found "Almost no attention to vision"; the policy leaned on joint state and task ID and memorised trajectories. Source: https://aadarshram.github.io/2025-07-29-Le-X-O. For this task that means the ball position must really vary between demonstrations, and the policy must be tested with the ball in a spot it has not memorised before anyone trusts it.

**Camera placement.** The official rule is "you should be able to do the task yourself by only looking at the camera images", and the LeRobot examples use a `wrist` plus a `top` camera. Source: https://huggingface.co/docs/lerobot/il_robots. The plate-wiping team listed "the arm occluding the overhead camera" among its main problems, and picking a sponge out of a bowl was the first hard step. Source: https://darthgera123.github.io/blogs/hackathon/robot. A forum case traced failures to a camera that had moved between training and evaluation and to a view occluded near the grasp. Source: https://discuss.huggingface.co/t/what-is-the-expected-success-rate-for-a-act-trained-model-using-50-episodes/178517. The dataset guide adds: at least 640x480, neutral steady lighting, no leader arm and no human limbs in frame. Source: https://huggingface.co/blog/lerobot-datasets

**Clearance next to neighbours and the wall.** Bin-picking research documents this directly: "an object at the border of a bin may not be graspable with a parallel jaw gripper because one finger collides with the bin wall" (https://arxiv.org/abs/2110.00992), and collision-free grasps can be impossible "due to object proximity to the bin walls or to other objects", for which the proposed fix is a push first (https://goldberg.berkeley.edu/pubs/CASE2018-mike-linear-push-policies-final.pdf). Standard grippers are "primarily designed for table-top grasping environments without constraints such as bin walls" (https://arxiv.org/abs/2504.01861).

**Rolling and clustering.** I found no source on round objects in a bowl for ACT. Several SO-101 ball checkpoints exist on the Hub (15, 50 and 60 episodes) with no evaluation reported. Source: https://huggingface.co/yokoyaman/act_so101_ball_60ep3. The physics is plain reasoning: a concave bowl sends every ball to the lowest point, so the target almost always touches a neighbour or the wall, which is exactly the clearance case above. **[unverified by any source]**

**Feeding detector output into the policy.**
- A Stanford course project (UR5, diffusion policy, wrist camera, LEGO among identical distractors) found that a four-number bounding box given as a low-dimensional input "was ignored", while the same box as an extra image mask channel gave 89% correct picks on a grid and 76% in piles, with 0% wrong-object picks. Policies with no target prompt "either selected arbitrary blocks or oscillated between nearby objects". The main remaining failure was reaching about 2 cm short at positions outside the training range. Source: https://cs224r.stanford.edu/projects/pdfs/Raul%20Garreta%20Tompson%20Joshua%20Alexander%20Bowden%20Swaroop%20Pal%20submission_416215691/cs224r_2026_final_report_point_and_pick.pdf
- One paper draws green and red boxes straight onto the RGB image for ACT on a UR5e (10 demonstrations per product, 70 to 100% success), but has no baseline without the boxes and contradicts itself on whether the boxes reach the network. Source: https://arxiv.org/html/2508.08748v1
- Google's MOO conditions a policy on a detector's object mask and reports about 79% success on unseen objects. Source: https://arxiv.org/abs/2303.00905
- None of these tests a fixed-colour target. Those papers need the marker because the target changes between episodes. Here the target is always the only blue thing, so colour already is the marker. I found no evidence that an overlay helps in that case, and it would have to be drawn into every training frame too. **[no direct evidence either way]**

---

## 3. Detecting a blue ball beside foil balls in a brown bowl

**Why foil is awkward for colour thresholds.** Reflected light is a mix of surface reflection (highlights) and body reflection (object colour). Source: https://www.ri.cmu.edu/publications/image-segmentation-and-reflection-analysis-through-color-2. On dielectrics the highlight is the colour of the light, so highlight pixels are usually found as low saturation plus high value, with hue undefined. Source: https://www.scitepress.org/Papers/2014/46801/46801.pdf. Crumpled foil is many small mirrors, so it shows white glints and also patches of whatever is nearby, including the blue ball. That last step is my inference from the reflection model; no source tested foil. **[unverified]** Reflections are a recognised cause of false detections in general. Source: https://arxiv.org/abs/1904.13273

**Option A: HSV threshold plus contour.** No training and no licence question. Brown and blue sit far apart in hue, so the bowl itself is easy. The failure cases are blue reflections in foil and any other blue object in view. Defences that follow from the above: restrict to a fixed bowl mask, require high saturation, require a minimum area and roundness, lock camera exposure and white balance, use diffuse light.

**Option B: small fine-tuned detector.**
- Current Ultralytics model: YOLO26, released 14 January 2026, AGPL-3.0, five sizes. A model trained on your own data stays AGPL-3.0. Source: https://roboflow.com/model-licenses/yolo26. Ultralytics lists open-source and academic projects as the free tier and requires an Enterprise licence if the project is not open-sourced. Source: https://www.ultralytics.com/license. RF-DETR is the Apache-2.0 alternative. Source: https://roboflow.com/model-licenses/yolo26
- Speed (640 px, COCO benchmark hardware): YOLO26n 38.9 ms on CPU with ONNX and 1.7 ms on a T4 with TensorRT; YOLO26s 87.2 ms and 2.5 ms. Source: https://docs.ultralytics.com/compare/yolo26-vs-yolov8
- Images: Roboflow's guidance is to fine-tune a COCO checkpoint on 50 to 100 labelled images (https://blog.roboflow.com/yolo-training/); Ultralytics' own ideal for production is 1,500 or more images per class (https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results). Training time for a few hundred images on the laptop GPU: I found no figure. **[estimate: minutes, not hours]**

**Option C: zero-shot open-vocabulary detectors.**
- YOLO-World runs inside the `ultralytics` package: `model.set_classes(["blue ball"])`, then predict; the custom vocabulary can be saved into a normal weights file. Source: https://docs.ultralytics.com/models/yolo-world/. The paper reports 35.4 AP on LVIS at 52 FPS on a V100. Source: https://arxiv.org/abs/2401.17270
- Grounding DINO-T: about 9.4 FPS in PyTorch and 42.6 FPS with TensorRT on an A100 at 800x1333, and about 1.1 FPS on an Orin NX. Source: https://arxiv.org/abs/2405.10300
- OWLv2: I found no latency figure. One paper times an OWL-family model at about 39 ms on an RTX 2080, but the checkpoint named is the older OWL-ViT. Source: https://arxiv.org/abs/2511.09955 **[unverified for OWLv2]**
- No source gave laptop-GPU or CPU speeds for any zero-shot detector, and none tested whether they separate "blue ball" from a blue reflection. **[unverified]** Their most useful role here is labelling training images for option B.

**"Visible enough to grasp".** The mechanical-search paper measures visibility as the ratio of visible mask area to full mask area, and decides graspability by a grasp-quality score above a threshold. Source: https://ar5iv.labs.arxiv.org/html/1903.01588. The practical equivalent is: blob area as a fraction of the area the ball shows when fully exposed at the same spot, plus roundness, held over several frames.

**Fusing two views.** I found no source on fusing two uncalibrated webcams for this. The recommendation in the last section is engineering judgement. **[no source]**

---

## 4. Stirring or rearranging to reveal hidden objects

**Names to use.** "Mechanical search" for finding and extracting a known target hidden by clutter; "interactive perception" for acting in order to see better; "singulation" for separating touching objects so one can be grasped.

**Key papers.**
- Danielczuk et al., ICRA 2019, "Mechanical Search: Multi-Step Retrieval of a Target Object Occluded by Clutter". Formalises the task for bins with 10 to 20 heaped objects; 15,000 simulated and 300 physical trials. On 15-object heaps the best policies succeeded in 94 to 98% of physical trials in 4.85 to 6.3 actions; a human supervisor got 98% in 3.06 actions. Adding a push action raised success in simulation from 90.3% to 93.3%. Sources: https://ai.stanford.edu/mech-search/multistep and https://ar5iv.labs.arxiv.org/html/1903.01588
- Danielczuk et al., IROS 2020, "X-Ray". A network predicts where the hidden target probably is and the robot acts to shrink that region: 82% success in simulation on 1,000 heaps of 15 objects, 15 points above the best baseline; on a real robot with 25-object heaps, 15% higher success with 17% fewer actions. Source: https://arxiv.org/abs/2004.09039
- Bohg et al., IEEE T-RO 2017, "Interactive Perception: Leveraging Action in Perception and Perception in Action". The survey that defines the field: interaction creates sensor signals that would not exist otherwise. Source: https://arxiv.org/abs/1604.03670
- Also useful: Novkovic et al., ICRA 2020, object finding by interaction, over 88% success on a real robot after training only in simulation (https://arxiv.org/abs/1911.07482); Zeng et al., IROS 2018, pushing learned purely because it makes later grasps succeed (https://arxiv.org/abs/1803.09956); and tactile retrieval from granular media, 54 to 63% success, which shows how hard search is without vision (https://arxiv.org/abs/2402.04536).

**When to stop.** The 2019 paper ends an attempt on any of: target grasped; no action above the confidence threshold; more than three pushes in a row; 2N actions for N objects; target knocked out of the workspace. Source: https://ar5iv.labs.arxiv.org/html/1903.01588. Note also that in that paper pushing is the fallback, chosen only when no grasp qualifies.

I found no paper on stirring with a gripper specifically; published work uses pushes, grasps and suction. **[no source for stirring]**

---

## 5. Handover and handshake

**Imitation-learning handover on low-cost arms.** I found none. The nearest are a dual-arm XLeRobot passing objects between its own two arms at a 2025 hackathon (https://seeed.cc/post/2025-embodied-ai-hackathon-recap) and Hub datasets named `so101_fill_cup_together` from a two-arm hackathon (https://huggingface.co/cl3mens). Neither involves a human hand. **[no direct prior work found]**

**What handover research says (Ortenzi et al., IEEE T-RO 2021).** The exchange starts when the receiver touches the object and ends when the giver lets go. "An early release can cause the object to fall"; most robots simply release when they sense a pull. Slower motion reduces stress on the receiver, and a robot that reacts quickly is rated better even if slightly less accurate. Suggested minimum metrics: success rate, total handover time, and rated fluency and trust. Source: https://ar5iv.labs.arxiv.org/html/2007.12952

**MediaPipe Hands today.**
- Latest `mediapipe` is 1.2.3, released 11 November 2026. Wheels are tagged `py3-none` for Windows x86-64 and ARM64, Linux x86-64 and ARM64 (glibc 2.28+), and macOS ARM64; classifiers list Python 3.9 to 3.14. So Python 3.12 is covered on both Windows and Linux. Source: https://pypi.org/project/mediapipe/
- Its dependencies are `absl-py~=2.3`, `certifi`, `numpy` (no version pin), `sounddevice~=0.5`, `flatbuffers~=25.9`, `opencv-contrib-python`, `matplotlib`. No protobuf. Source: https://pypi.org/pypi/mediapipe/json
- The old `mp.solutions` API is gone "from MediaPipe version 0.10.30 onwards"; the maintainers say to use MediaPipe Tasks. Drawing helpers returned in 0.10.32. Source: https://github.com/google-ai-edge/mediapipe/issues/6192
- Older releases are the ones with the famous conflicts: 0.10.20 pinned `protobuf>=4.25.3,<5` and refused numpy 2. Sources: https://github.com/google-ai-edge/mediapipe/issues/5679 and https://github.com/cansik/mediapipe-numpy2. Do not pin an old version to get `mp.solutions` back: LeRobot 0.7.0 requires `numpy>=2.0.0,<2.3.0` and Python 3.12 or newer. Source: https://raw.githubusercontent.com/huggingface/lerobot/v0.7.0/pyproject.toml
- The live conflict is OpenCV. LeRobot 0.7.0 depends on `opencv-python-headless>=4.9.0,<4.14.0` and MediaPipe on `opencv-contrib-python`. OpenCV's packaging notes say "SELECT ONLY ONE OF THEM. Do not install multiple different packages in the same environment", because all share the `cv2` namespace. Sources: same pyproject link and https://pypi.org/project/opencv-python-headless/
- Speed: HandLandmarker (full) averages 17.12 ms on CPU and 12.27 ms on GPU on a Pixel 6 phone. No laptop figure found, but a phone CPU already manages about 58 frames per second. Source: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index
- Open palm: the Gesture Recognizer's built-in classes include `Open_Palm` alongside `Closed_Fist`, `Thumb_Up` and others; it returns nothing when no hand is seen. Source: https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python. Whether `Open_Palm` tells palm-up from palm-down is not documented. **[unverified]**

**Safety in plain words.**
- ISO/TS 15066 is a technical specification, not a full standard. For robots that may touch people it limits force and pressure per body region, based on where pain starts. Source: https://blog.robotiq.com/what-is-isots-15066
- It separates two cases. Clamping ("quasi-static"): the body part is trapped against something fixed and cannot move away. Free impact ("transient"): the body part can move away. "The general rule is a transient impact can apply a force two times higher". The example given for a hand crushed against a table is 135 N, and 270 N free. Source: https://blog.robotiq.com/collaborative-robot-risk-assessment-impact. A search result quoting a reproduction of the specification's body-model table gives 140 N for hands and fingers; the paper is most likely https://arxiv.org/abs/2311.13814, which I did not open. I could not read the specification itself, so both the exact figure and that attribution are **[unverified]**.
- The specification derives a speed limit from those forces: allowed energy is force squared over twice the tissue stiffness, so lighter and slower robots are allowed more. Source: https://arxiv.org/abs/2304.14059
- Measured pain-onset limits for the hand are lower when the contact is narrow: for pinching, the forefinger pad tolerates 150 N on a blunt surface but only 70 N/cm2 on a semi-sharp one; the palm 150 N and 60 N/cm2. Most ISO limits are "preliminary, since they are based on unverified data". Source: https://www.frontiersin.org/articles/10.3389/frobt.2021.667818/full

**SO-101 gripper force.**
- No published pinch-force figure exists that I could find. **[none found]**
- Servo data: STS3215 stall torque is 19.5 kg.cm at 7.4 V (rated load 5 kg.cm) and 30 kg.cm for the 12 V version (rated 10 kg.cm, stall current 2.7 A). Sources: https://www.dfrobot.com/product-2961.html and https://www.dfrobot.com/product-2962.html
- LeRobot caps the gripper in software: `Max_Torque_Limit` 500 ("50% of max torque to avoid burnout"), `Protection_Current` 250, `Overload_Torque` 25. It also offers `max_relative_target` to clip how far one command may move a joint. Source: https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/robots/so_follower/so_follower.py
- **[estimate, jaw length assumed 8 cm, not measured]** 19.5 kg.cm is 1.91 N.m, so about 24 N at the jaw tip at stall and about 12 N with the 50% cap; the 12 V servo gives about 37 N and 18 N. That is well under the 135 to 150 N blunt-contact figures. But 20 N on a quarter of a square centimetre of printed plastic edge is 80 N/cm2, above the 60 to 70 N/cm2 pinch limits above. Force is not the problem; narrow edges and the gap between the jaws are.

**Earlier handshake robots.** A review groups the field and finds people and robots synchronise through the phases of a handshake (https://ar5iv.labs.arxiv.org/html/2102.07193). Disney Research showed that arm stiffness and timing change how human-like and how pleasant a robot handshake feels (https://la.disneyresearch.com/wp-content/uploads/mura_IJRR.pdf). Human handshakes last about 3.6 seconds across reach, contact and return (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8996188/). All used compliant arms or sensorised hands; none used a hobby servo gripper.

---

## 6. Pitch framing

**Real problems it maps to.**
- Mechanical search in warehouses, homes and shops. The founding paper opens with robots "operating in unstructured environments such as warehouses, homes, and retail centers". Source: https://ar5iv.labs.arxiv.org/html/1903.01588
- Bin picking in clutter. Amazon's Sparrow picks from "crowded totes" and still "struggles with items that have loose or complex packaging". Sources: https://www.vision-systems.com/embedded/article/14287150/amazon-testing-robotic-arm-that-identifies-individual-products and https://techcrunch.com/2022/11/10/amazon-debuts-sparrow-a-new-bin-picking-robot-arm/
- Assistive handover: fetching a named item and placing it in a person's hand.
- Pharmacy and lab picking: finding the right item among look-alikes.

**Numbers that can go on a slide.**
1. Order picking is estimated at up to 55% of total warehouse operating cost. Source: https://pure.eur.nl/ws/portalfiles/portal/46713708/DesignandControl_2007.pdf
2. Amazon deployed its one-millionth robot in July 2025, and Sparrow handles about 65% of a catalogue of over 100 million items. Sources: https://techcrunch.com/2025/07/01/amazon-deploys-its-one-millionth-robot-releases-generative-ai-model and https://www.therobotreport.com/rbr50-company-2023/amazon-looks-to-automate-holy-grail-with-sparrow-robot-arm/
3. A human finds a hidden object in a 15-object heap in 3.06 actions; the best automatic policies in the same 2019 study needed 4.85 to 6.3. Source: https://ar5iv.labs.arxiv.org/html/1903.01588
4. About 150,000 Americans could benefit from a wheelchair-mounted robot arm, yet "only a few hundred" are in use; in one care-home study an arm cut nursing help by 40%, from 3.7 to 2.8 hours a day. Source: https://ar5iv.labs.arxiv.org/html/2104.03892
5. Pharmacy dispensing errors pool at about 1.6% worldwide; one hospital robot cut caught errors from 0.204% to 0.044%. Sources: https://pubmed.ncbi.nlm.nih.gov/37848350/ and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9526262/

Market-size figures for picking robots differ by a factor of about six between publishers (USD 652 million in 2023 from one, USD 3.7 billion in 2024 from another). Do not quote one. Sources: https://www.imarcgroup.com/piece-picking-robots-market and https://marketresearchfuture.com/reports/piece-picking-robot-market-36515

**Earlier LeRobot and SO-101 hackathon projects.** What I could identify: plate wiping, conveyor waste sorting, tic-tac-toe (twice), pens into a holder, shorts folding and kitchen cleaning, an "AI bartender", two arms passing objects, and a cooking robot. Sources as cited above, plus https://aojedao.github.io/Portfolio/projects/lerobot-hackathons. None searches for a hidden object and none hands an object to a person. The official list of 30 winners from the 2025 worldwide hackathon gives only team names (for example leChef, ez_2_ai, ML6, LeCyborg), so I could not check them. **[novelty unverified]** Source: https://huggingface.co/api/datasets/maringetxway/all-winners/tree/main

**How to differentiate.** The distinctive part is the loop "cannot see it, so act to see it, then look again", followed by a decision that depends on a person. Picking a coloured object alone is the most common demo there is.

---

## What this means for the build

**Architecture**
1. Keep the state machine. Make it the only thing that decides, and let it decide only while the arm is parked at home, so both cameras have a clear view and perception does not compete with the policy for the laptop.
2. Learn PICK. Consider learning GIVE. Script or replay everything else: MIX is a fixed circular path, PUT DOWN goes to a fixed spot, SHAKE is a short up-and-down. `lerobot-replay` replays a recorded episode on the robot (https://huggingface.co/docs/lerobot/il_robots), so one good teleoperated take of each is enough. This is my recommendation from the evidence in section 1, not something a source tested on this task.
3. One home pose. Every demonstration starts there and ends there with a one-second hold. A scripted move returns to home after every skill, including after failures.
4. The awkward handoff is PICK to GIVE or PUT DOWN, because the arm arrives holding a ball. Record those demonstrations starting from where PICK really ends, ball in gripper.
5. Completion is decided outside the policy: a timeout per skill, then a perception check from home (is blue still in the bowl? is the gripper reading not fully closed?). Limit retries to two.
6. Load all checkpoints at start-up. Check in the first hour whether LeRobot 0.7.0 lets one process hold two policies and one robot; if not, the fallback is one process per skill started by the state machine, at the cost of load time.

**Data budget (100 to 200 episodes)**
- PICK: 80 to 120. Spread over ball positions and over the number of other balls, including the fuller bowl used after stirring. About 10 per distinct arrangement.
- GIVE: 30 to 40 if learned. If the hand must be held in a marked zone, it can be replayed instead and those episodes go to PICK.
- Keep 10 to 15 held-out arrangements for testing and score them separately from trained ones.
- Training: ACT fits in 2 to 6 GB, so it trains on the laptop. On Hugging Face Jobs an L4 costs USD 0.80 an hour and an A10G USD 1.00 (https://huggingface.co/docs/hub/jobs-pricing), and ACT on 50 episodes takes about 1 to 2 hours there, so the 30 EUR covers roughly 15 to 30 runs. Set `--steps` to 5 to 10 epochs, not the 100,000 default, and set a job timeout.

**Physical setup**
- Wide, shallow, matte bowl, fixed to the table. Steep walls and a loose bowl are the two avoidable failures.
- Blue ball: soft and grippy, clearly smaller than the open jaw. Foil balls: larger than the jaw gap is better, so they cannot be picked by mistake.
- Keep the total at five or six balls for the "buried" case. More makes the post-stir grasp harder for no gain in the story.
- One camera high above looking into the bowl, framing the handover zone too. The second on the wrist. Lock exposure and white balance, fix both mounts, mark their positions with tape.
- Constant diffuse light. Foil changes appearance with every lighting change, for the detector and the policy alike.

**Detection**
- Start with a saturated-blue threshold inside a hand-drawn bowl mask on the top camera, with minimum area and roundness, stable over about 5 of 8 frames. Calibrate "fully visible" area once per bowl position.
- If foil reflections cause false positives within the first hour of testing, switch to a fine-tuned YOLO26n: 150 to 200 frames from the actual cameras, including foil-only bowls as negatives. The repository is then AGPL-3.0, which is fine for an open hackathon repo.
- Two views: top camera decides "graspable". Wrist camera, from a fixed look-in pose, can only upgrade "not visible" to "visible", never the reverse.
- Stirring: stir one loop, return home, look. Stop after three or four rounds and say so out loud; a clean "I could not find it" is a better demo than endless stirring.

**Handover and handshake**
- Install MediaPipe in its own virtual environment and run it as a separate process that sends "hand present, open, position" to the state machine. That removes the OpenCV clash and keeps the policy loop at 30 Hz.
- Hand held in the air, not resting on the table, so it can move away if touched. The arm comes from above, stops a centimetre or two over the palm, and opens. No pull detection is needed and nothing closes on the person.
- Handshake: the person grips the robot's closed gripper and the robot makes a small slow up-and-down. The gripper never closes on a hand. Aim for 3 to 4 seconds in total.
- Leave LeRobot's 50% gripper torque cap in place, set `max_relative_target`, round or pad the jaw edges, slow the approach, abort if the hand leaves the zone, and keep one person on the stop key. A team member should be the hand, not a judge, unless it has worked ten times in a row.

**Pitch**
- Lead with: "Robots can pick what they can see. Ours goes looking for what it cannot see, and then hands it to a person."
- Name the research field (mechanical search, interactive perception) and be open that decisions are classical and the grasp is learned. That is the design the research and earlier teams point to.
- Show one number of your own: successes out of N on held-out arrangements.

---

## Unverified

- No source shows per-skill ACT plus a state machine on an SO-100 or SO-101, working or failing. The case for it is assembled from adjacent evidence.
- The tic-tac-toe team's "hybrid would have been better" remark comes from a search snippet; the page returned 403.
- Whether LeRobot 0.7.0 can hold several policies, or several rollout contexts on one robot, in a single process. Not documented in what I read.
- The inference docs I read are the current online version; I confirmed interactive sessions and custom strategies in the 0.7.0 release notes (https://github.com/huggingface/lerobot/releases/tag/v0.7.0) but `RolloutController` is not named there.
- Round objects rolling and clustering in a bowl, and stirring as a search action: no source found. Reasoning only.
- Whether an overlay or mask helps when the target is always the same colour: no evidence either way.
- Foil reflecting the blue ball strongly enough to fool a threshold: inferred from reflection physics, not tested by any source.
- Zero-shot detector speed on a laptop GPU or CPU, OWLv2 latency, and whether such detectors separate a blue ball from its reflection: not found.
- YOLO fine-tuning time on an RTX 4070 laptop for a few hundred images: estimate.
- ISO/TS 15066 hand limit: 135 N from one secondary source, 140 N from another. The specification itself was not read. Its values are described as preliminary.
- SO-101 pinch force: computed from servo stall torque with an assumed 8 cm jaw. Measure the real jaw, or press the gripper on a kitchen scale.
- Whether the team's follower uses 7.4 V or 12 V servos; both versions exist.
- MediaPipe latency on a laptop CPU (only a Pixel 6 figure exists), and whether `Open_Palm` distinguishes palm-up.
- MediaPipe 1.2.3 dependency list and wheel tags were read from PyPI through a summarising tool; confirm with `pip install mediapipe` in a clean environment before relying on them.
- Novelty: 30 winning projects from the 2025 worldwide hackathon could not be inspected.
- Several fetches failed (403 or 404): the Seeed cooking-robot recap, a Medium post on a failed SO-100 project, and a candy-bar ACT guide. Their content is not used except where a search snippet is cited and flagged.
