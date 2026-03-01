<script lang="ts">
  export let value = '';
  export let rows = 4;
  export let readonly = false;
  export let placeholder = '';
  export let className = '';
</script>

<textarea bind:value {rows} {readonly} {placeholder} class={`textarea ${className}`} />

<style>
  .textarea {
    width: 100%;
    border-radius: var(--radius);
    border: 1px solid hsl(var(--border));
    background: white;
    padding: 0.7rem 0.75rem;
    font-size: 0.9rem;
    line-height: 1.45;
    resize: vertical;
    outline: none;
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
  }

  .textarea:focus {
    border-color: hsl(var(--primary));
    box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.15);
  }
</style>
