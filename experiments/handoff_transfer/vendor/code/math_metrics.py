class MetricsWorker:
    def sd_install_monitor(self):
        from vllm.utils import jit_monitor
        if not hasattr(jit_monitor,"sd_events"):
            jit_monitor.sd_events=0
            old=jit_monitor._handle_jit_event
            def counted(**kwargs):
                jit_monitor.sd_events+=1
                return old(**kwargs)
            jit_monitor._handle_jit_event=counted
        return True

    def sd_stats(self):
        import torch
        from vllm.utils import jit_monitor
        d=getattr(self.model_runner,"drafter",None)
        t=getattr(d,"transfer_teacher",None)
        return dict(jit_events=getattr(jit_monitor,"sd_events",0),device_name=torch.cuda.get_device_name(),device_uuid=str(torch.cuda.get_device_properties(0).uuid),allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved(), teacher_graph_captures=getattr(t,"graph_captures",0), teacher_eager_prefills=getattr(t,"eager_prefills",0))

