"""与安装后的oridynamic命令使用相同入口；旧无子命令形式仍兼容。"""
from oridynamic.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
