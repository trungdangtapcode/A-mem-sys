import * as SliderPrimitives from "@radix-ui/react-slider";
import * as React from "react";
import { cn } from "@/lib/utils";

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitives.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitives.Root>
>(({ className, ...props }, ref) => (
  <SliderPrimitives.Root
    ref={ref}
    className={cn("relative flex w-full touch-none select-none items-center", className)}
    {...props}
  >
    <SliderPrimitives.Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-white/10">
      <SliderPrimitives.Range className="absolute h-full rounded-full bg-primary" />
    </SliderPrimitives.Track>
    <SliderPrimitives.Thumb className="block h-5 w-5 rounded-full border border-white/20 bg-white shadow-[0_6px_18px_-10px_rgba(255,255,255,1)] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/80 disabled:pointer-events-none disabled:opacity-50" />
  </SliderPrimitives.Root>
));
Slider.displayName = SliderPrimitives.Root.displayName;

export { Slider };
