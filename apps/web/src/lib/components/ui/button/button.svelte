<script lang="ts">
  export let type: 'button' | 'submit' | 'reset' = 'button';
  export let disabled = false;
  export let variant: 'default' | 'secondary' | 'destructive' | 'ghost' | 'success' = 'default';
  export let className = '';
</script>

<button {type} {disabled} class={`btn ${variant} ${className}`}>
  <slot />
</button>

<style>
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    border-radius: var(--radius);
    border: 1px solid transparent;
    padding: 0.55rem 0.95rem;
    font-size: 0.875rem;
    font-weight: 700;
    cursor: pointer;
    transition: filter 0.15s ease, transform 0.1s ease;
  }

  .btn:active {
    transform: translateY(1px);
  }

  .btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .default {
    background: hsl(var(--primary));
    color: hsl(var(--primary-foreground));
  }

  .success {
    background: #166534;
    color: #ffffff;
    border-color: #14532d;
  }

  .secondary {
    background: hsl(var(--muted));
    color: hsl(var(--foreground));
  }

  .destructive {
    background: #b42318;
    color: #ffffff;
    border-color: #912018;
  }

  .ghost {
    background: transparent;
    color: hsl(var(--foreground));
    border-color: hsl(var(--border));
  }

  .btn:hover:not(:disabled) {
    filter: brightness(1.03);
  }
</style>
