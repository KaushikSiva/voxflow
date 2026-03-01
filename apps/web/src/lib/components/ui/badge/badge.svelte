<script lang="ts">
  export let variant: 'ok' | 'warn' | 'idle' | 'error' = 'idle';
  export let className = '';
</script>

<span class={`badge ${variant} ${className}`}>
  <slot />
</span>

<style>
  .badge {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    border: 1px solid transparent;
    padding: 0.2rem 0.55rem;
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: none;
  }

  .ok {
    background: #f1f1f1;
    color: #111111;
    border-color: #d8d8d8;
  }

  .warn {
    background: #f3f3f3;
    color: #1a1a1a;
    border-color: #d4d4d4;
  }

  .idle {
    background: #f5f5f5;
    color: #2b2b2b;
    border-color: #d9d9d9;
  }

  .error {
    background: #fff0f1;
    color: #9e1a22;
    border-color: #f0c6ca;
  }
</style>
