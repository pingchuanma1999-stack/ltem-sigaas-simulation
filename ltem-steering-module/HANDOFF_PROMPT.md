# 交给 Claude Code 的任务说明

**用法**
1. 把整个 `ltem-steering-module/` 文件夹放进你的光学模拟软件仓库，比如放在仓库根目录，或者 `third_party/` 下面。
2. 在这个仓库里打开 Claude Code，把下面分隔线以下的英文部分整段粘贴给它。

如果放的位置不是仓库根目录，把提示里的路径改一下。

**它会做的事**
1. 先读 README，再摸清你的代码结构。
2. 提出集成方案，等你确认后才动手。
3. 根据你软件的语言选择实现方式：
   - Python 项目：直接用 NumPy 版；
   - JS/TS 项目：直接用 JS 版；
   - 其他语言：按规范移植，并用标准答案验证。
4. 物理核心保持原样。坐标和单位的换算单独写在一层适配代码里。
5. 加上标准答案测试和物理自检测试。
6. 最后汇报改了什么、怎么调用、测试结果。

**需要你拍板的地方**
- 你软件里的坐标系怎么对应到这里的实验室坐标系：+X 朝入射口那面墙，+Y 朝后，+Z 朝上，原点在针尖。
- 参数放在哪里配置。
- 界面上要不要显示光斑图。

---

I'm adding a self-contained physics module, `ltem-steering-module/`, to this repository (my optics simulation software). Please integrate it.

**What it is.** The module models the pump spot of a non-local LTEM experiment on our neaspec THz s-SNOM. The pump is steered by an external Thorlabs KS1-Z8 mirror mount with two Z812 actuators. Given each actuator's push-rod travel in µm, the module computes:
- where the focused spot moves on the sample (exact chief-ray trace);
- how the spot's shape and peak intensity change (scalar diffraction with the parabolic mirror's aperture);
- the input beam diameter that gives the smallest spot;
- peak-intensity curves versus travel.

The module contains:
- a JS reference implementation, `js/ltem_optics.mjs`;
- a NumPy port with the same API, `python/ltem_optics.py`;
- golden reference outputs, `tests/golden.json`, which both implementations reproduce;
- a full algorithm specification in `README.md`;
- the interactive page that uses the same core, `reference/ltem-steering.html`, so you can see the intended behaviour.

**Steps**

1. Read `ltem-steering-module/README.md` completely, especially:
   - "Conventions": lab frame, actuator signs, units, array layout, intensity normalisations;
   - "API";
   - "Algorithm".

   Run both test suites once to confirm the module works here:
   ```
   node ltem-steering-module/js/test_golden.mjs
   python -m pytest ltem-steering-module/python -q
   ```
   Use whichever runtimes are available; NumPy and pytest are needed for the second command.

2. Explore this repository. Find out:
   - the language and frameworks;
   - how optical elements, beams/rays, sources and results are represented;
   - which coordinate frames and units are used;
   - how parameters are configured;
   - how results are displayed;
   - how tests run;
   - where a physics module like this belongs.

3. **Before writing code**, give me a short integration plan and wait for my OK. Cover:
   - where the module goes;
   - which implementation you will use and why;
   - how its inputs and outputs map onto our data structures (frames, units, sign conventions);
   - what you will expose in our config, API or UI;
   - which tests you will add.

   Ask me if any of these is ambiguous, in particular:
   - how our coordinate frame relates to the module's lab frame;
   - whether our software already models the steering mirror or the parabolic mirror in a way that conflicts with this module.

4. Choose the implementation by the language of this codebase:
   - **Python**: vendor `python/ltem_optics.py` as is. Its only runtime dependency is NumPy.
   - **JavaScript/TypeScript**: vendor `js/ltem_optics.mjs`. You may add type declarations or convert it to TypeScript, but do not change the numerics.
   - **Anything else**: port it following the README's algorithm section, with the two reference implementations as a guide. The port is done when it reproduces `tests/golden.json` within the tolerances stored in that file.

5. Keep the physics core pure and numerically unchanged. Do not change any of these:
   - grid sizes, thresholds or iteration limits;
   - the soft aperture;
   - the float32 storage of the intensity map;
   - the ±1 sign parameters.

   The golden tests pin all of them. Put every conversion between our conventions and the module's into **one** thin adapter layer and test it:
   - coordinate frame;
   - units (geometry in mm; travel, spot shifts and spot maps in µm);
   - actuator naming (`yaw` = lower, horizontal Z812; `pitch` = upper, vertical Z812; positive = push rod extends = KDC101 reading increases).

6. Tests to add to our suite:
   - **Golden tests.** Load `tests/golden.json` and compare against it, as `js/test_golden.mjs` and `python/test_golden.py` do, with the tolerances stored in the file.
   - **Sanity checks through our adapter**, so that a sign or frame mistake in the adapter is caught. With the default parameters:
     - lower actuator +1 µm → spot moves 0.848 µm left (toward −X, away from the port wall) and 0.321 µm toward the front (−Y);
     - upper actuator +1 µm → spot moves 0.737 µm right and 0.923 µm toward the front;
     - aligned spot: Strehl (`peak`) > 0.99, long axis along the plane of incidence, aspect ratio between 1.7 and 2.3.

7. Expose the parameters with the README's defaults:
   - geometry: `R`, `theta`, `beta`, `alpha`, `dist`, `L`, `portX`, `foldY`;
   - beam and aperture: `lam`, `D`, `NA`, `zLow`;
   - actuator orientation: `yawSide`, `pitchSide`.

   `D='auto'` (the beam diameter for the smallest spot) costs 50–90 ms. The module caches it per parameter set; keep that behaviour, or cache it in our layer.

8. Performance:
   - A 'fine' spot takes ~6–120 ms, 'fast' ~3–55 ms and 'trend' ~2–42 ms.
   - In interactive code, use 'fast' while the user is dragging and 'fine' when idle.
   - Use 'trend' for sweeps.
   - Do not block a UI thread on 'fine' at large travel.

9. Do not edit `tests/golden.json`. It may only be regenerated with `node tests/make_golden.mjs`, and only if I explicitly ask for a physics change.

10. When done, summarise:
    - the files you added or changed;
    - how to call the module from our code (a short example);
    - where the parameters live;
    - test results;
    - every convention mapping you decided on.

Do not port the reference page's UI unless I ask. It is there to show expected behaviour: open `reference/ltem-steering.html` in a browser, with internet access for three.js.
