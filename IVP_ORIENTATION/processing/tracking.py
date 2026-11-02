import math

class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # Dictionary of ID: (cx, cy)
        self.max_distance = max_distance

    def update(self, centroids):
        if len(centroids) == 0:
            return {}

        updated_objects = {}
        used_ids = set()

        for (cx, cy) in centroids:
            matched_id = None
            min_dist = self.max_distance

            # Find closest existing object
            for obj_id, (ox, oy) in self.objects.items():
                if obj_id in used_ids:
                    continue
                dist = math.hypot(cx - ox, cy - oy)
                if dist < min_dist:
                    min_dist = dist
                    matched_id = obj_id

            if matched_id is not None:
                updated_objects[matched_id] = (cx, cy)
                used_ids.add(matched_id)
            else:
                # Register new object
                updated_objects[self.next_object_id] = (cx, cy)
                self.next_object_id += 1

        self.objects = updated_objects
        return self.objects