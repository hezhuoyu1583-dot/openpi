# 公司网络代理配置指南

## ⚠️ 重要提醒

在安装任何 Python 包或系统包之前，**必须配置公司代理**。

## 🔧 代理信息

```
代理地址：http://proxy.hs.com:3128
```

## 📝 配置方式

### 方式 1：临时配置（当前会话）

```bash
export http_proxy=http://proxy.hs.com:3128
export https_proxy=http://proxy.hs.com:3128
export HTTP_PROXY=http://proxy.hs.com:3128
export HTTPS_PROXY=http://proxy.hs.com:3128
```

### 方式 2：永久配置（推荐）

编辑 `~/.bashrc` 或 `~/.zshrc`：

```bash
# 添加以下行到文件末尾
export http_proxy=http://proxy.hs.com:3128
export https_proxy=http://proxy.hs.com:3128
export HTTP_PROXY=http://proxy.hs.com:3128
export HTTPS_PROXY=http://proxy.hs.com:3128
export no_proxy=localhost,127.0.0.1,*.internal,*.company.com
```

然后使配置生效：

```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

### 方式 3：pip 配置文件

编辑 `~/.pip/pip.conf`：

```ini
[global]
proxy = http://proxy.hs.com:3128
```

### 方式 4：apt 配置

编辑 `/etc/apt/apt.conf.d/proxy.conf`（需要 sudo）：

```
Acquire::http::Proxy "http://proxy.hs.com:3128";
Acquire::https::Proxy "http://proxy.hs.com:3128";
```

## ✅ 验证代理配置

```bash
# 检查当前代理设置
echo $http_proxy
echo $https_proxy

# 测试连接
curl -I https://pypi.org/pypi

# pip 测试
pip config list

# apt 测试（需要 sudo）
sudo apt-get update
```

## 🚀 安装前快速检查

每次安装前运行：

```bash
# 确保代理已设置
if [ -z "$http_proxy" ]; then
  export http_proxy=http://proxy.hs.com:3128
  export https_proxy=http://proxy.hs.com:3128
fi

# 验证
echo "当前代理: $http_proxy"

# 现在可以安装
bash install_sim_env_python310.sh
```

## 📋 常见问题

### Q: 代理一直需要配置怎么办？

**A:** 使用方式 2（永久配置）将代理写入 ~/.bashrc。

### Q: pip install 仍然超时？

**A:** 
```bash
# 增加超时时间
pip install --default-timeout=1000 package_name

# 或使用国内镜像（如果允许）
pip install -i https://mirrors.aliyun.com/pypi/simple/ package_name
```

### Q: apt-get 仍然不工作？

**A:**
```bash
# 使用 sudo 时需要保留环境变量
sudo -E apt-get install package

# 或编辑 /etc/apt/apt.conf.d/ 中的配置文件
```

## 🎯 快速命令集

```bash
# 一键设置代理（临时）
export http_proxy=http://proxy.hs.com:3128 https_proxy=http://proxy.hs.com:3128

# 安装仿真环境
bash install_sim_env_python310.sh

# 验证安装
examples/libero/.venv/bin/python -c "import mujoco, robosuite; print('✓ OK')"
```

## 📌 部署检查清单

部署到新机器时：

- [ ] 配置代理（临时或永久）
- [ ] 验证代理连接 (`curl -I https://pypi.org/`)
- [ ] 运行 `sudo apt-get update`
- [ ] 运行 `pip install --upgrade pip`
- [ ] 运行安装脚本 `bash install_sim_env_python310.sh`

---

**最后更新：** 2026-07-01  
**相关脚本：** `install_sim_env_python310.sh`
