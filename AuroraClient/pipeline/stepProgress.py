"""Step-based progress for long single-scan tools.

Batch tools report progress per scan on the shared `progress_group` channel. Tools that
work on one scan but still take long (random landmarking, saving landmarks, mesh cleanup,
snapping, semi-landmarks, ...) have no per-scan loop, so they report their stages instead,
through the same channel and the same message shape:

    current = the step now running (1-based; 0 = starting), total = number of steps,
    progress = fraction of steps already finished, custom_message = what the step does.

Stages take very different times (an ANTs registration vs. a file write), so an ETA
learned from earlier steps would be wrong. `timing: 'elapsed'` tells the client to show
time elapsed instead of time left. engineJobs records how long each step took in the job
history (~/.aurora/job_history.jsonl), so typical step ratios can later be learned there
to estimate time left for these tools too.
"""

from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync


def send_step_progress(step, total_steps, message, scan_name=''):
    """Report that step `step` of `total_steps` (`message`) has started. Best effort."""
    channel_layer = get_channel_layer()
    if channel_layer is None or total_steps <= 0:
        return
    step = max(0, min(int(step), int(total_steps)))
    try:
        async_to_sync(channel_layer.group_send)(
            'progress_group',
            {
                'type': 'send_progress',
                'progress': max(0.0, (step - 1) / total_steps) if step > 0 else 0,
                'scan_name': scan_name,
                'custom_message': message,
                'total': int(total_steps),
                'current': step,
                'timing': 'elapsed',
            }
        )
    except Exception as e:  # progress is cosmetic; never fail the tool over it
        print(f"send_step_progress: could not send progress ({e})")


class StepProgress:
    """Small convenience for a view: `steps = StepProgress(4, filename)`, then
    `steps.start()` once and `steps(2, 'Removing islands')` at each stage."""

    def __init__(self, total_steps, scan_name='', title=''):
        self.total = int(total_steps)
        self.scan_name = scan_name or ''
        self.title = title

    def start(self, title=None):
        send_step_progress(0, self.total, title or self.title, self.scan_name)

    def __call__(self, step, message):
        send_step_progress(step, self.total, message, self.scan_name)
