<script lang="ts">
  export let className = '';
</script>

<section class={`card ${className}`}>
  <slot />
</section>

<style>
  .card {
    border-radius: calc(var(--radius) + 4px);
    border: 1px solid hsl(var(--border));
    background: hsl(var(--card));
    color: hsl(var(--card-foreground));
    padding: 1.2rem;
    box-shadow: 0 18px 40px -30px rgba(20, 30, 40, 0.35);
  }
</style>
