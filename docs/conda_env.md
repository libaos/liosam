# Conda 环境（统一入口）

本仓库的运行环境（如 `conda_envs/`、`miniforge3/`）默认不进 git（见 `.gitignore`），但我们用**可复现的 env 定义文件**把依赖统一起来，避免“机器上装了一堆但不知道哪套能跑”。

## RandLA / Torch（GPU）环境：`randla39`

- 默认前缀：`conda_envs/randla39`（历史兼容：`conda_envs/randlanet39`）
- ROS launch 里可通过 `RANDLA_ENV_PREFIX` / `ROS_PYTHON_EXEC` 覆盖该前缀/解释器。

### 推荐用法（不使用 venv）

```bash
cd /mysda/w/w/lio_ws
source tools/ros_py39/setup.bash
echo "$RANDLA_ENV_PREFIX"
echo "$ROS_PYTHON_EXEC"
```

### 需要 `conda activate` 时（可选）

```bash
cd /mysda/w/w/lio_ws
source miniforge3/etc/profile.d/conda.sh
export CONDARC="$(pwd)/.condarc"
conda activate randla39
```

注：如果你在脚本里用了 `set -u`（nounset），建议不要直接 `conda activate`，改用 `source tools/ros_py39/setup.bash` 或临时 `set +u`。

### 创建/重建（可选）

如果你后续补齐了 env 定义文件（例如 `conda/envs/randla39.yml`），可以用脚本创建：

```bash
cd /mysda/w/w/lio_ws
bash tools/conda/create_randla_env.sh
```

### 渠道配置（可选）

脚本会优先使用 `CONDARC` 环境变量；否则若存在 `conda/condarc.yml` 会自动使用它（默认 `conda-forge` + 镜像）。
