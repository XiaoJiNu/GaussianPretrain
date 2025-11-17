#!/usr/bin/env python3
"""环境验证脚本"""

import sys
import subprocess

def check_python():
    """检查 Python 版本"""
    print("=" * 60)
    print("1. 检查 Python 版本")
    print("=" * 60)
    version = sys.version_info
    print(f"当前 Python 版本: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor == 8:
        print("✅ Python 3.8 - 正确")
    else:
        print("❌ 需要 Python 3.8")
        return False
    return True

def check_pytorch():
    """检查 PyTorch 和 CUDA"""
    print("\n" + "=" * 60)
    print("2. 检查 PyTorch 和 CUDA")
    print("=" * 60)
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"CUDA 是否可用: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"    显存: {mem_total:.1f} GB")

        if torch.__version__.startswith('1.9'):
            print("✅ PyTorch 1.9.x - 正确")
        else:
            print("⚠️  推荐使用 PyTorch 1.9.1")

        if torch.version.cuda == '11.1':
            print("✅ CUDA 11.1 - 正确")
        else:
            print("⚠️  推荐使用 CUDA 11.1")

        return True
    except Exception as e:
        print(f"❌ PyTorch 检查失败: {e}")
        return False

def check_mmcv():
    """检查 mmcv"""
    print("\n" + "=" * 60)
    print("3. 检查 mmcv-full")
    print("=" * 60)
    try:
        import mmcv
        print(f"mmcv 版本: {mmcv.__version__}")
        if mmcv.__version__.startswith('1.3'):
            print("✅ mmcv-full 1.3.x - 正确")
        else:
            print("⚠️  推荐使用 mmcv-full 1.3.11")
        return True
    except Exception as e:
        print(f"❌ mmcv 检查失败: {e}")
        return False

def check_mmdet3d():
    """检查 mmdet3d"""
    print("\n" + "=" * 60)
    print("4. 检查 mmdet3d")
    print("=" * 60)
    try:
        import mmdet3d
        print(f"mmdet3d 版本: {mmdet3d.__version__}")
        if mmdet3d.__version__.startswith('0.17'):
            print("✅ mmdet3d 0.17.x - 正确")
        else:
            print("⚠️  推荐使用 mmdet3d 0.17.3")
        return True
    except Exception as e:
        print(f"❌ mmdet3d 检查失败: {e}")
        return False

def check_plugin():
    """检查 mmdet3d_plugin"""
    print("\n" + "=" * 60)
    print("5. 检查 mmdet3d_plugin")
    print("=" * 60)
    try:
        import projects.mmdet3d_plugin
        print("✅ mmdet3d_plugin 加载成功")
        return True
    except Exception as e:
        print(f"❌ mmdet3d_plugin 加载失败: {e}")
        print("请运行: python setup.py develop")
        return False

def check_gaussian_rasterization():
    """检查 diff-gaussian-rasterization"""
    print("\n" + "=" * 60)
    print("6. 检查 diff-gaussian-rasterization (关键)")
    print("=" * 60)
    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        print("✅ diff-gaussian-rasterization 加载成功")

        # 测试基本功能
        import torch
        if torch.cuda.is_available():
            settings = GaussianRasterizationSettings(
                image_height=100,
                image_width=100,
                tanfovx=1.0,
                tanfovy=1.0,
                bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
                scale_modifier=1.0,
                viewmatrix=torch.eye(4, dtype=torch.float32, device="cuda"),
                projmatrix=torch.eye(4, dtype=torch.float32, device="cuda"),
                sh_degree=3,
                campos=torch.zeros(3, dtype=torch.float32, device="cuda"),
                prefiltered=False,
                debug=False
            )
            print("✅ 可以创建 GaussianRasterizationSettings")
        return True
    except Exception as e:
        print(f"❌ diff-gaussian-rasterization 检查失败: {e}")
        print("请运行:")
        print("  cd projects/mmdet3d_plugin/ops/diff-gaussian-rasterization")
        print("  python setup.py develop")
        return False

def check_dependencies():
    """检查其他依赖"""
    print("\n" + "=" * 60)
    print("7. 检查其他关键依赖")
    print("=" * 60)

    packages = [
        ('numpy', '1.19'),
        ('scipy', None),
        ('nuscenes', None),
        ('mmdet', '2.14'),
        ('mmseg', '0.14'),
    ]

    all_ok = True
    for pkg_name, expected_version in packages:
        try:
            if pkg_name == 'nuscenes':
                pkg = __import__('nuscenes')
            else:
                pkg = __import__(pkg_name)

            version = getattr(pkg, '__version__', 'unknown')
            status = "✅" if expected_version is None or version.startswith(expected_version) else "⚠️"
            print(f"{status} {pkg_name}: {version}", end="")
            if expected_version:
                print(f" (推荐: {expected_version}.x)")
            else:
                print()
        except ImportError:
            print(f"❌ {pkg_name}: 未安装")
            all_ok = False

    return all_ok

def check_gpu_memory():
    """检查 GPU 显存"""
    print("\n" + "=" * 60)
    print("8. GPU 显存建议")
    print("=" * 60)
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i} 显存: {mem_total:.1f} GB")

                if mem_total >= 32:
                    print(f"  ✅ 可以使用 voxel_size=0.6 (最精细)")
                elif mem_total >= 24:
                    print(f"  ✅ 可以使用 voxel_size=0.6 + FP16")
                elif mem_total >= 16:
                    print(f"  ⚠️  建议使用 voxel_size=0.8 + FP16")
                else:
                    print(f"  ⚠️  建议使用 voxel_size=1.0 + FP16 + 减少采样")
        return True
    except:
        return True

def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "GaussianPretrain 环境验证" + " " * 23 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    checks = [
        check_python,
        check_pytorch,
        check_mmcv,
        check_mmdet3d,
        check_plugin,
        check_gaussian_rasterization,
        check_dependencies,
        check_gpu_memory,
    ]

    results = []
    for check in checks:
        try:
            results.append(check())
        except Exception as e:
            print(f"检查过程出错: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)

    if all(results[:6]):  # 前6项是必需的
        print("✅ 所有必需组件检查通过！可以开始训练和测试。")
        return 0
    else:
        print("❌ 部分组件检查失败，请按照上述提示修复。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
