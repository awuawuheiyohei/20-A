"""
调度器 —— 11:30 / 14:30 自动触发 + 提供手动 trigger API。

使用 APScheduler，cron 表达式触发。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Any, Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger


log = logging.getLogger("triggers")


@dataclass
class TradingDayScheduler:
    """
    A 股交易日定时调度器。

    Usage:
        sched = TradingDayScheduler()
        sched.add_morning_close(analyze_morning_close)
        sched.add_afternoon_close(analyze_afternoon_close)
        sched.start()    # 阻塞
    """

    morning_time: str = "11:30"
    afternoon_time: str = "14:30"
    scheduler: BlockingScheduler = field(default_factory=BlockingScheduler)
    morning_job_id: str = "morning_close"
    afternoon_job_id: str = "afternoon_close"

    def __post_init__(self):
        # APScheduler 时区配置（系统时区一般是 Asia/Shanghai）
        # 如果 tzdata 没装就用本地时间
        try:
            import tzdata  # noqa
        except ImportError:
            pass

    def _parse_time(self, hhmm: str) -> tuple[int, int]:
        h, m = hhmm.split(":")
        return int(h), int(m)

    def add_morning_close(self, fn: Callable[[], Any]) -> None:
        h, m = self._parse_time(self.morning_time)
        self.scheduler.add_job(
            fn,
            CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
            id=self.morning_job_id,
            name=f"Morning close @ {self.morning_time}",
            replace_existing=True,
        )
        log.info("scheduled morning_close @ %s", self.morning_time)

    def add_afternoon_close(self, fn: Callable[[], Any]) -> None:
        h, m = self._parse_time(self.afternoon_time)
        self.scheduler.add_job(
            fn,
            CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
            id=self.afternoon_job_id,
            name=f"Afternoon close @ {self.afternoon_time}",
            replace_existing=True,
        )
        log.info("scheduled afternoon_close @ %s", self.afternoon_time)

    def start(self) -> None:
        """阻塞运行。按 Ctrl+C 退出。"""
        self.scheduler.start()

    def stop(self) -> None:
        self.scheduler.shutdown(wait=False)

    def list_jobs(self) -> list[dict]:
        jobs = []
        for job in self.scheduler.get_jobs():
            nrt = getattr(job, "next_run_time", None)
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": str(nrt) if nrt else "(未启动)",
            })
        return jobs

    # 手动触发（调试 / 测试用）
    def trigger_morning_now(self) -> None:
        job = self.scheduler.get_job(self.morning_job_id)
        if job and job.func:
            job.func()

    def trigger_afternoon_now(self) -> None:
        job = self.scheduler.get_job(self.afternoon_job_id)
        if job and job.func:
            job.func()


# ---------------------------------------------------------------------- #
#  Self-test：列出当前 jobs
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sched = TradingDayScheduler()

    def _dummy():
        print(f"[{datetime.now()}] dummy triggered")

    sched.add_morning_close(_dummy)
    sched.add_afternoon_close(_dummy)
    for j in sched.list_jobs():
        print(j)
    print("(jobs are added but scheduler not started; just demonstration)")