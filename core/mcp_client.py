import asyncio
import os
import shutil
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_12306_ENTRY = PROJECT_ROOT / "12306-mcp" / "build" / "index.js"


def _get_node_command() -> str:
    configured = os.getenv("TICKET_12306_NODE")
    if configured:
        return str(Path(configured).expanduser())
    home = Path.home()
    nvm_root = home / ".nvm" / "versions" / "node"
    if nvm_root.exists():
        candidates = sorted(nvm_root.glob("v*/bin/node"), reverse=True)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return shutil.which("node") or "node"


def _get_12306_server_params() -> StdioServerParameters:
    entry = Path(os.getenv("TICKET_12306_MCP_ENTRY", str(DEFAULT_12306_ENTRY))).expanduser()
    if not entry.exists():
        raise FileNotFoundError(
            "未找到 12306 MCP 构建文件："
            f"{entry}。请先在项目根目录执行：cd 12306-mcp && npm install && npm run build"
        )
    return StdioServerParameters(command=_get_node_command(), args=[str(entry)])


async def call_12306_tool(tool_name: str, arguments: dict) -> str:
    """
    通用调用 12306 MCP 工具的函数。
    tool_name: 工具名，见下方 SUPPORTED_TOOLS
    arguments: 工具参数字典
    返回: 工具返回的文本内容
    """
    async with stdio_client(_get_12306_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return "\n".join(
                block.text for block in result.content if block.type == "text"
            )


# ---- 以下是对每个工具的便捷封装 ----

async def get_current_date() -> str:
    """获取当前日期 (Asia/Shanghai, yyyy-MM-dd)"""
    return await call_12306_tool("get-current-date", {})


async def get_stations_in_city(city: str) -> str:
    """查询某个城市的所有火车站"""
    return await call_12306_tool("get-stations-code-in-city", {"city": city})


async def get_station_code_of_citys(citys: str) -> str:
    """获取一个或多个城市的代表站点代码，多个城市用 | 分隔"""
    return await call_12306_tool("get-station-code-of-citys", {"citys": citys})


async def get_station_code_by_names(station_names: str) -> str:
    """通过具体站名获取站点代码，多个站名用 | 分隔"""
    return await call_12306_tool("get-station-code-by-names", {"stationNames": station_names})


async def get_tickets(
    date: str,
    from_station: str,
    to_station: str,
    train_filter: str = "",
    earliest_start: int = 0,
    latest_start: int = 24,
    sort_flag: str = "",
    sort_reverse: bool = False,
    limited_num: int = 0,
    format: str = "text",
) -> str:
    """
    查询车票
    date: 日期 yyyy-MM-dd
    from_station: 出发站（中文站名/城市名/station_code）
    to_station: 到达站
    train_filter: 车型筛选，G=高铁 D=动车 Z=直达 T=特快 K=快速 O=其他 F=复兴号 S=智能动车
    earliest_start: 最早出发小时 (0-24)
    latest_start: 最晚出发小时 (0-24)
    sort_flag: 排序方式 startTime/arriveTime/duration
    limited_num: 限制返回数量，0=不限
    format: 输出格式 text/csv/json
    """
    args = {
        "date": date,
        "fromStation": from_station,
        "toStation": to_station,
        "trainFilterFlags": train_filter,
        "earliestStartTime": earliest_start,
        "latestStartTime": latest_start,
        "sortFlag": sort_flag,
        "sortReverse": sort_reverse,
        "limitedNum": limited_num,
        "format": format,
    }
    return await call_12306_tool("get-tickets", args)


async def get_interline_tickets(
    date: str,
    from_station: str,
    to_station: str,
    middle_station: str = "",
    show_wz: bool = False,
    train_filter: str = "",
    earliest_start: int = 0,
    latest_start: int = 24,
    sort_flag: str = "",
    sort_reverse: bool = False,
    limited_num: int = 10,
    format: str = "text",
) -> str:
    """查询中转/联程车票"""
    args = {
        "date": date,
        "fromStation": from_station,
        "toStation": to_station,
        "middleStation": middle_station,
        "showWZ": show_wz,
        "trainFilterFlags": train_filter,
        "earliestStartTime": earliest_start,
        "latestStartTime": latest_start,
        "sortFlag": sort_flag,
        "sortReverse": sort_reverse,
        "limitedNum": limited_num,
        "format": format,
    }
    return await call_12306_tool("get-interline-tickets", args)


async def get_train_route_stations(
    train_code: str, depart_date: str, format: str = "text"
) -> str:
    """查询某趟列车的经停站信息"""
    return await call_12306_tool(
        "get-train-route-stations",
        {"trainCode": train_code, "departDate": depart_date, "format": format},
    )


# ---- 测试 ----
if __name__ == "__main__":
    async def main():
        # 示例: 查明天北京到上海的高铁票
        date = await get_current_date()
        print(f"当前日期: {date}")

        result = await get_tickets(
            date="2026-05-14",
            from_station="北京",
            to_station="上海",
            train_filter="G",
            limited_num=0,
        )
        print(result)

    asyncio.run(main())
