# consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

class ProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add("progress_group", self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("progress_group", self.channel_name)

    async def send_progress(self, event):
        progress = event['progress']
        scan_name = event['scan_name']
        thetype = event['type']
        current = event['current']
        total = event['total']
        custom_message = event['custom_message']
        payload = {
            'type': thetype,
            'progress': progress,
            'scan_name': scan_name,
            'total': total,
            'current': current,
            'custom_message': custom_message,
        }
        for extra_key in (
            'run_id',
            'train_loss',
            'val_metrics',
            'best_checkpoint',
            'run_status',
        ):
            if extra_key in event:
                payload[extra_key] = event[extra_key]
        await self.send(text_data=json.dumps(payload))

class MemoryConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        await self.channel_layer.group_add('memory_group', self.channel_name)
    
    async def disconnect(self, close_code):
        await self.channel_layer.group_discard('memory_group', self.channel_name)
    
    async def send_memory(self, event):
        response_data = {
            'program_memory_gb': event['app_memory_usage'],
            'system_memory_gb': event['system_memory_usage'],
        }
        
        # Include swap memory data if available
        if event.get('has_swap', False):
            response_data['swap_memory_gb'] = event['swap_memory_usage']
            response_data['has_swap'] = True
        else:
            response_data['has_swap'] = False
            
        await self.send(text_data=json.dumps(response_data))
        
