"""
SORT: A Simple, Online and Realtime Tracker
Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Adaptado para el proyecto de conteo vehicular
"""

from __future__ import print_function

import numpy as np
from filterpy.kalman import KalmanFilter

def linear_assignment(cost_matrix):
    """Realiza asignación lineal usando el algoritmo Húngaro"""
    try:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))
    except ImportError:
        # Fallback si scipy no está disponible
        from munkres import Munkres
        m = Munkres()
        indices = m.compute(cost_matrix)
        return np.array(indices)


def iou_batch(bb_test, bb_gt):
    """
    Calcula IOU entre dos conjuntos de bounding boxes.
    bb_test: detecciones actuales [N, 4] en formato [x1,y1,x2,y2]
    bb_gt: tracks existentes [M, 4] en formato [x1,y1,x2,y2]
    Returns: matriz [N, M] con valores IOU
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)  


def convert_bbox_to_z(bbox):
    """
    Convierte bbox [x1,y1,x2,y2] al formato [x,y,s,r] donde:
    x,y: centro
    s: escala/área
    r: aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    # escala es el área
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Convierte estado [x,y,s,r] de vuelta a bbox [x1,y1,x2,y2]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    Representa el estado interno de un objeto rastreado usando Kalman Filter.
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Inicializa tracker con bbox [x1, y1, x2, y2]
        """
        # Filtro de Kalman con 7 estados [x,y,s,r,vx,vy,vs] y 4 observaciones [x,y,s,r]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.  # Incertidumbre en observación de escala
        self.kf.P[4:,4:] *= 1000. # Incertidumbre alta en velocidades iniciales
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Actualiza el estado con la observación bbox [x1,y1,x2,y2]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Avanza el estado y devuelve la predicción de bbox
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Devuelve el bbox estimado actual [x1,y1,x2,y2]
        """
        return convert_x_to_bbox(self.kf.x)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Parámetros:
          max_age: Máximo de frames sin detección antes de eliminar track
          min_hits: Mínimo de detecciones consecutivas antes de confirmar track
          iou_threshold: Mínimo IOU para asociar detección con track
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Parámetros:
          dets: array numpy de detecciones en formato [[x1,y1,x2,y2,score], ...]
        
        Devuelve array similar pero con el track_id en la última columna:
          [[x1,y1,x2,y2,track_id], ...]
        
        NOTA: El número de objetos devueltos puede diferir del número de detecciones.
        """
        self.frame_count += 1
        
        # Obtener predicciones de todos los trackers existentes
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Asociar detecciones con trackers usando IOU
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Actualizar trackers con matches
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Crear nuevos trackers para detecciones no asociadas
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1)) # +1 para que IDs empiecen en 1
            i -= 1
            # Remover trackers muertos
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Asocia detecciones con trackers existentes usando IOU.
    
    Returns:
        matched: array Nx2 con índices [det_idx, trk_idx]
        unmatched_detections: array con índices de detecciones sin match
        unmatched_trackers: array con índices de trackers sin match
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filtrar matches con IOU bajo
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
